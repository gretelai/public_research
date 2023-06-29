import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

POSSIBLE_STAR_RATINGS = [1, 5]
PROJECT_PATH = Path(__file__).parent.resolve()
PROMPT_TEMPLATE = "Product Name: {product_title}\nNumber of Stars: {star_rating}\nReview: {review_body}"


def fetch_and_clean_dataset(subset: str = "Apparel_v1_00") -> pd.DataFrame:
    """Fetch and clean the Amazon US Reviews dataset from the Hugging Face Hub.

    Currently, the only cleaning step is to replace line break tag with newlines.
    Read about the dataset here https://huggingface.co/datasets/amazon_us_reviews.

    Args:
        subset: Subset of the dataset. Defaults to "Apparel_v1_00".

    Returns:
        Cleaned dataset as a pandas DataFrame.
    """
    print(f"Fetching amazon_us_reviews {subset} dataset")
    dataset = load_dataset("amazon_us_reviews", subset, split="train")
    df = dataset.to_pandas()
    df["review_num_words"] = df["review_body"].str.split().apply(len)
    for col in ["product_title", "review_body"]:
        df[col] = df[col].str.replace(r"<br\s/>", "\n\n", regex=True)
        df[col] = df[col].str.replace("&#34;", '"')
    print(f"Number of reviews in dataset: {len(df)}")
    return df


def select_groups(
    df: pd.DataFrame, min_words: int = 10, max_words: int = 50
) -> tuple[pd.DataFrame, pd.core.groupby.DataFrameGroupBy]:
    """Group reviews by product_id and select based on number of words and star ratings.

    We select records with at least one review for each star rating and with review
    length between min_words and max_words. Words are defined as space separated tokens.

    Args:
        df: Full review dataset with product ids, reviews, and star ratings.
        min_words: Minimum number of words in a review. Defaults to 10.
        max_words: Maximum number of words in a review.. Defaults to 50.

    Returns:
        Tuple with elements (selected records, selected records grouped by product_id).
    """
    print(f"Selecting records with {min_words} < review_num_words < {max_words}")
    select_mask = (df["review_num_words"] < max_words) & (df["review_num_words"] > min_words)
    print("Grouping records by product_id")

    # Keep groups with at least one example of star_rating in POSSIBLE_STAR_RATINGS.
    df_groupby_product_id = (
        df[select_mask]
        .groupby("product_id")
        .filter(lambda group: all([(group["star_rating"] == n).sum() >= 1 for n in POSSIBLE_STAR_RATINGS]))
        .groupby("product_id")
    )
    print(f"Number of groups: {df_groupby_product_id.ngroups}")

    return df[select_mask], df_groupby_product_id


def create_training_review_pairs(df_groupby: pd.core.groupby.DataFrameGroupBy, rng: np.random.mtrand.RandomState):
    """Create review pairs for fine-tuning Gretel GPT.

    We pair reviews with 1 and 5 star ratings for each product_id, where the second review is the target
    and the first review is the reference. We randomly select which star rating to use as the reference.

    Example review pair:

        Product Name: Nickelodeon Teenage Mutant Ninja Turtles Michelangelo Romper Shell and Headpiece
        Number of Stars: 5
        Review: My son looked adorable in this costume. At 13 months the newborn fit him just fine.
                He is on the short side though.

        Product Name: Nickelodeon Teenage Mutant Ninja Turtles Michelangelo Romper Shell and Headpiece
        Number of Stars: 1
        Review: When I think infant I think small not a infant size 6 months. Only good thing is he
                can wear it for Halloween.

    Args:
        df_groupby: Grouped records by product_id.
        rng: RandomState object for random number generation.

    Returns:
        DataFrame where each row is a review pair.
    """
    review_pair_list = []

    for _, df_group in tqdm(df_groupby, total=df_groupby.ngroups, desc="Creating review pairs"):
        select_idx = rng.randint(2)
        reference_num_stars = POSSIBLE_STAR_RATINGS[select_idx]
        target_num_stars = POSSIBLE_STAR_RATINGS[1 - select_idx]

        # Select the most helpful review for each star rating.
        reviews = {
            num_stars: df_group.query(f"star_rating == {num_stars}")
            .sort_values("helpful_votes", ascending=False)
            .iloc[0]
            for num_stars in POSSIBLE_STAR_RATINGS
        }

        review_pair_list.append(
            f"{PROMPT_TEMPLATE.format(**reviews[reference_num_stars])}\n\n"
            f"{PROMPT_TEMPLATE.format(**reviews[target_num_stars])}"
        )

    return pd.DataFrame(review_pair_list, columns=["text"])


def create_conditional_prompt_test_set(
    df: pd.DataFrame,
    group_names: list[str],
    rng: np.random.mtrand.RandomState,
    num_samples: int = 100,
) -> pd.DataFrame:
    """Create test set of conditional prompts.

    The conditional prompts are simply review pairs with an empty second review,
    which the fine-tuned model will generate.

    Args:
        df: Product review dataset with product ids, reviews, and star ratings.
        group_names: List of all the product id groups.
        rng: RandomState object for random number generation.
        num_samples: Number of conditional prompts. Defaults to 100.

    Returns:
        DataFrame where each row is a conditional prompt.
    """
    print(f"Creating {num_samples} conditional prompts")

    # For each star rating, we select reviews that were not in the training group.
    df["star_rating"].isin(POSSIBLE_STAR_RATINGS)
    not_in_group_mask = ~df["product_id"].isin(group_names)

    samples = []
    for star_rating in POSSIBLE_STAR_RATINGS:
        star_mask = df["star_rating"] == star_rating
        samples.append(df[not_in_group_mask & star_mask].sample(num_samples // 2, replace=False, random_state=rng))
    samples = pd.concat(samples)

    conditional_prompt_list = []
    no_review_template = "Product Name: {product_title}\nNumber of Stars: {star_rating}\nReview:"
    for _, row in samples.iterrows():
        target_num_stars = (
            POSSIBLE_STAR_RATINGS[0] if row["star_rating"] == POSSIBLE_STAR_RATINGS[1] else POSSIBLE_STAR_RATINGS[1]
        )
        conditional_prompt_list.append(
            f"{PROMPT_TEMPLATE.format(**row)}\n\n"
            f"{no_review_template.format(product_title=row['product_title'], star_rating=target_num_stars)}"
        )

    return pd.DataFrame(conditional_prompt_list, columns=["text"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-subset", type=str, default="Video_Games_v1_00", help="Subset of Amazon dataset to use.")
    parser.add_argument("--max-training-samples", type=int, default=25000, help="Maximum number of samples to save.")
    parser.add_argument("--min-words", type=int, default=10, help="Minimum number of words in a review.")
    parser.add_argument("--max-words", type=int, default=50, help="Maximum number of words in a review.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-conditional-samples", type=int, default=100, help="Number of testing samples to save.")
    parser.add_argument("--output-path", type=Path, default=PROJECT_PATH / "data", help="Save the output files here.")
    args = parser.parse_args()

    assert len(POSSIBLE_STAR_RATINGS) == 2, "only two star ratings are supported for swapping"

    start_time = time.time()

    df = fetch_and_clean_dataset(args.data_subset)
    df_select, df_groupby = select_groups(df, args.min_words, args.max_words)

    rng = np.random.RandomState(args.seed)
    df_training = create_training_review_pairs(df_groupby, rng)

    print(f"Writing file with {args.max_training_samples} training samples")
    df_training = df_training.sample(min(len(df_training), args.max_training_samples), replace=False, random_state=rng)
    df_training.to_csv(
        args.output_path / f"training_product_review_pairs_{args.data_subset}.csv.gz", index=False, compression="gzip"
    )

    group_names = list(df_groupby.groups.keys())
    df_conditional = create_conditional_prompt_test_set(df_select, group_names, rng, args.num_conditional_samples)
    df_conditional.to_csv(
        args.output_path / f"conditional_prompts_{args.data_subset}.csv.gz", index=False, compression="gzip"
    )

    end_time = time.time()
    print(f"Finished in {(end_time - start_time)/60:.2f} minutes")
