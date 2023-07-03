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
assert len(POSSIBLE_STAR_RATINGS) == 2, "only two star ratings are supported for swapping"


def fetch_and_clean_dataset(subset: str = "Video_Games_v1_00") -> pd.DataFrame:
    """Fetch and clean the Amazon US Reviews dataset from the Hugging Face Hub.

    Currently, the only cleaning step is to replace line break tag with newlines.
    Read about the dataset here https://huggingface.co/datasets/amazon_us_reviews.

    Args:
        subset: Subset of the dataset. Defaults to "Video_Games_v1_00".

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
        df[col] = df[col].str.replace("&#45;", "-")
        df[col] = df[col].str.replace("&#36;", "$")
    print(f"Number of reviews in dataset: {len(df)}")
    return df


def select_and_group_reviews(
    df: pd.DataFrame, min_words: int = 10, max_words: int = 50
) -> tuple[pd.DataFrame, pd.core.groupby.DataFrameGroupBy]:
    """Select reviews based on number of words and star rating and group by product_id.

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
    select_mask &= df["star_rating"].isin([1, 5])

    # Keep groups with at least one example of star_rating in POSSIBLE_STAR_RATINGS.
    print("Grouping records by product_id")
    df_groupby_product_id = (
        df[select_mask]
        .groupby("product_id")
        .filter(lambda group: all([(group["star_rating"] == n).sum() >= 1 for n in POSSIBLE_STAR_RATINGS]))
        .groupby("product_id")
    )
    print(f"Total number of groups: {df_groupby_product_id.ngroups}")

    return df[select_mask], df_groupby_product_id


def create_review_pairs_helpful_votes(
    df_groupby: pd.core.groupby.DataFrameGroupBy,
    number_of_pairs: int,
    rng: np.random.mtrand.RandomState,
):
    """Create review pairs using helpful_votes as the pair selection metric.

    We pair reviews with 1 and 5 star ratings for each product_id, where the second review is the target
    and the first review is the reference. We randomly select which star rating to use as the reference.
    If there are multiple reviews with the same star rating, we select the first review in a list sorted
    by helpful_votes. Use this function if you only have access to a CPU, as it is much faster than
    create_training_review_pairs_cosine_similarity.

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
        number_of_pairs: Number of review pairs to create.
        rng: RandomState object for random number generation.

    Returns:
        DataFrame where each row is a review pair.
    """
    review_pair_list = []
    group_name_list = list(df_groupby.groups.keys())
    group_name_list = [group_name_list[i] for i in rng.choice(number_of_pairs, size=number_of_pairs, replace=False)]

    for group_name in tqdm(group_name_list, total=len(group_name_list), desc="Creating review pairs"):
        df_group = df_groupby.get_group(group_name)

        # Randomly select the reference and target star ratings.
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


def create_review_pairs_cosine_similarity(
    df: pd.DataFrame,
    df_groupby: pd.core.groupby.DataFrameGroupBy,
    number_of_pairs: int,
    rng: np.random.mtrand.RandomState,
    chunk_size: int = 100,
    batch_size: int = 32,
):
    """Create review pairs using the cosine similarity as the pair selection metric.

    Note: This function requires pytorch and sentence-transformers to be installed.
    GPU or MPS is recommended for faster processing. Processing 20,000 review pairs
    takes about 20 minutes on an M2 MacBook Pro with device = "mps".

    Args:
        df: Full review dataset with product ids, reviews, and star ratings.
        df_groupby: Groupby object, grouped by product_id.
        number_of_pairs: Number of groups to select from df_groupby.
        rng: Random number generator.
        chunk_size: Number of groups to process at a time. Defaults to 100.
        batch_size: Sentence transformer encoding batch size. Defaults to 32.

    Returns:
        The review pairs in a DataFrame with column name "text".
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim as cosine_similarity

    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    device = "mps" if torch.has_mps else "cuda" if torch.has_cuda else "cpu"

    # Randomly select number_of_pairs groups from df_groupby.
    group_name_list = list(df_groupby.groups.keys())
    group_name_list = [group_name_list[i] for i in rng.choice(number_of_pairs, size=number_of_pairs, replace=False)]

    with tqdm(total=len(group_name_list), desc="Creating review pairs") as progress_bar:
        # Iterate over chunks of groups.
        review_pair_list = []
        for i_chunk in np.arange(len(group_name_list), step=chunk_size):
            # Get the reviews for the current chunk.
            group_names = group_name_list[i_chunk : i_chunk + chunk_size]
            df_group_chunk = df.loc[np.concatenate([df_groupby.groups[name] for name in group_names])]

            # Encode all reviews in current chunk for each star rating.
            index_star_ratings = []
            embeddings_star_ratings = []
            for i in range(len(POSSIBLE_STAR_RATINGS)):
                index_star_ratings.append(df_group_chunk.query(f"star_rating == {POSSIBLE_STAR_RATINGS[i]}").index)
                embeddings_star_ratings.append(
                    embedding_model.encode(
                        df_group_chunk.loc[index_star_ratings[i], "review_body"].tolist(),
                        device=device,
                        batch_size=batch_size,
                    )
                )

            # Create review pair for each group.
            for name in group_names:
                index_group = []
                embeddings_group = []
                for i in range(len(POSSIBLE_STAR_RATINGS)):
                    group_mask = np.isin(index_star_ratings[i], df_groupby.groups[name])
                    index_group.append(index_star_ratings[i][group_mask])
                    embeddings_group.append(embeddings_star_ratings[i][group_mask])

                # Compute cosine similarity matrix between reviews with opposite star rating.
                cos_sim_matrix = cosine_similarity(*embeddings_group)

                # Select the review pair with the highest cosine similarity.
                df_group = df_groupby.get_group(name)
                i_nearest, j_nearest = np.unravel_index(cos_sim_matrix.argmax(), cos_sim_matrix.shape)
                reviews = [df_group.loc[index_group[0][i_nearest]], df_group.loc[index_group[1][j_nearest]]]

                # Randomly select which review to use as the reference.
                select_idx = rng.randint(2)
                review_pair_list.append(
                    f"{PROMPT_TEMPLATE.format(**reviews[select_idx])}\n\n"
                    f"{PROMPT_TEMPLATE.format(**reviews[1 - select_idx])}"
                )

                progress_bar.update(1)

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
    parser.add_argument("--min-words", type=int, default=10, help="Minimum number of words in a review.")
    parser.add_argument("--max-words", type=int, default=50, help="Maximum number of words in a review.")
    parser.add_argument("--max-training-samples", type=int, default=20000, help="Maximum number of samples to save.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-conditional-samples", type=int, default=100, help="Number of test examples to save.")
    parser.add_argument("--output-path", type=Path, default=PROJECT_PATH / "data", help="Save the output files here.")
    parser.add_argument(
        "--pair-with-cos-sim",
        action="store_true",
        help="Use cosine similarity as the pair selection metric. "
        "Note: You'll need to install pytorch and sentence-transformers to use this option.",
    )
    args = parser.parse_args()

    start_time = time.time()

    df = fetch_and_clean_dataset(args.data_subset)
    df_select, df_groupby = select_and_group_reviews(df, args.min_words, args.max_words)
    number_of_reviews = min(df_groupby.ngroups, args.max_training_samples)

    rng = np.random.RandomState(args.seed)
    if args.pair_with_cos_sim:
        print("Review pair metric: cosine similarity")
        df_training = create_review_pairs_cosine_similarity(df_select, df_groupby, number_of_reviews, rng)
    else:
        print("Review pair metric: helpful votes")
        df_training = create_review_pairs_helpful_votes(df_groupby, number_of_reviews, rng)

    print(f"Writing file with {len(df_training)} samples for fine tuning")
    df_training = df_training.sample(min(len(df_training), args.max_training_samples), replace=False, random_state=rng)
    df_training.to_csv(args.output_path / f"training_review_pairs_{args.data_subset}.csv.gz", index=False)

    group_names = list(df_groupby.groups.keys())
    df_conditional = create_conditional_prompt_test_set(df_select, group_names, rng, args.num_conditional_samples)
    df_conditional.to_csv(args.output_path / f"conditional_prompts_{args.data_subset}.csv.gz", index=False)

    end_time = time.time()
    print(f"Finished in {(end_time - start_time)/60:.2f} minutes")
