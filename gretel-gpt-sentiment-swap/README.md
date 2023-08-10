<img src="../assets/gretel_icon.jpg" height="85" width="85" align="left" style="margin-right: 0px"/>

# Gretel GPT Sentiment Swap

This repo contains the code and notebooks for the Gretel GPT Sentiment Swap blog post.

## 🎭 Use case overview

The code in this folder demonstrates how to use the [Gretel GPT API](https://docs.gretel.ai/reference/synthetics/models/gretel-gpt) to fine tune and prompt a large language model (LLM) to swap the sentiments of product reviews. Given a product review with a particular sentiment (positive or negative), our fine-tuned model will generate a new review with the opposite sentiment.

## 💾 Dataset for fine-tuning

We use Gretel GPT to fine tune an LLM using subsets of the [Amazon Customer Reviews](https://huggingface.co/datasets/amazon_us_reviews) dataset. For our task, the important columns of this dataset are 

- `product_id`
- `product_title`
- `star_rating`
- `review_body`

We use `star_rating` as a proxy for sentiment, where 5 star is positive and 1 star is negative. We group the records by `product_id` to build a dataset with both positive and negative reviews for thousands of products. We then pass this dataset to Gretel GPT to fine tune an LLM to generate review pairs (with opposite sentiment) for each product. 


## 📝 Code descriptions

- [create_dataset.py](./create_dataset.py): Uses the Amazon Customer Reviews dataset to build the fine-tuning and conditional prompt datasets.
- [fine_tune_with_gretel_gpt.py](./fine_tune_with_gretel_gpt.py): Fine tunes a Gretel GPT model on the fine-tuning dataset. You can view model training progress in the [Gretel Cloud console](https://console.gretel.ai/projects).
- [generate_sentiment_swapped_reviews.py](./generate_sentiment_swapped_reviews.py): Uses the fine-tuned Gretel GPT model to generate sentiment swapped reviews for each product in the conditional prompt dataset.
- [gretel_gpt_sentiment_swap.ipynb](./gretel_gpt_sentiment_swap.ipynb): Optional Jupyter notebook that can be used in place of `generate_sentiment_swapped_reviews.py` to generate sentiment swapped reviews with the fine-tuned model.

## 🤖 Example Gretel GPT generations

We fine tuned two Gretel GPT models on two different subsets of the Amazon Customer Reviews dataset: `Video_Games_v1_00` and  `Apparel_v1_00`. Here are some example generations for each (bold text indicates the model generations):

### Video_Games_v1_00

---

> Product Name: Just Dance 4<br>
> Number of Stars: 5<br>
> Review: It is a really good game. My daughter enjoys playing and working out with it
>
> Product Name: Just Dance 4<br>
> Number of Stars: 1<br>
> Review: **`This game is so old. It was made in 2012, and now that I bought it, they are making a new one that is way better. This one has a really low budget. I suggest not to buy it, get the newer one that is way better.`**

---

> Product Name: Red Nunchuck Wired Controller Compatible with Nintendo Wii<br>
> Number of Stars: 1<br>
> Review: This product seemed like it would be great and a great price. It quit working after the second use. Complete junk.
>
> Product Name: Red Nunchuck Wired Controller Compatible with Nintendo Wii<br>
> Number of Stars: 5<br>
> Review: **`This is an excellent wired nunchuck. I purchased it to use with my old Wii and I haven't had any problems with it. Works just as it should.`**

---

### Apparel_v1_00

---

> Product Name: Little Boys Paw Patrol Toddler Hat<br>
> Number of Stars: 5<br>
> Review: Great hat and great fit...well made. Purchased for my nephew and he loves it..
>
> Product Name: Little Boys Paw Patrol Toddler Hat<br>
> Number of Stars: 1<br>
> Review:  **`The hat I received had a big rip in the back and no tags inside to put the child's name. It is also a different hat than what is pictured in the ad.`**

---

> Product Name: Disney Girls' 2 Piece Minnie Knit Pant Set<br>
> Number of Stars: 1<br>
> Review: As with all the Minnie Mouse clothes my daughter got for Christmas, this shrunk from a 4t to not fitting my 2 year old daughter at all.  So disappointed with Disney clothing.
>
> Product Name: Disney Girls' 2 Piece Minnie Knit Pant Set<br>
> Number of Stars: 5<br>
> Review: **`The product is as described, and as expected. It looks great on my daughter, and the material is of great quality. Very happy with the product.`**

---

## 👩‍🔬 Steps to reproduce our results

### Installation

To run this example, clone this repository and the install the dependencies in `requirements.txt`, which include the [Gretel Python SDK](https://github.com/gretelai/gretel-python-client).

Clone this repo and cd into the `gretel-gpt-sentiment-swap` directory:
```bash
git clone https://github.com/gretelai/public_research.git
cd public_research/gretel-gpt-sentiment-swap
```

If you only want to clone the contents of the gretel-gpt-sentiment-swap folder, you can use GitHub’s sparse-checkout functionality:

```bash
git clone --no-checkout https://github.com/gretelai/public_research.git
cd public_research

git sparse-checkout init --cone
git sparse-checkout set gretel-gpt-sentiment-swap
git checkout main

cd gretel-gpt-sentiment-swap
```

Next, install the dependencies (preferably within a virtual Python environment):

```bash
python -m pip -r requirements.txt
```
> Note: The results presented here were generated using Python 3.10.12.

You will also need to [install Jupyter](https://jupyter.org/install) if you want to run the notebook.


### Create an account on the Gretel platform

If you haven't already, create an account on the [Gretel platform](https://console.gretel.ai/login). The free developer tier comes with 60 credits (5 hours of compute) per month, which is enough to fine tune one model on the `Video_Games_v1_00` data subset!

To run the code in this repo, you'll need a Gretel API key, which can be found in the [account settings page](https://console.gretel.ai/users/me/key) of your Gretel console. You will be prompted to enter your API key when you run the code. Alternatively, you can set an environment variable called `GRETEL_API_KEY` to your API key.

### Build the fine-tuning dataset

The [create_dataset.py](./create_dataset.py) script performs the following steps to create the fine-tuning dataset:
- download the given subset (passed as a command-line argument) of the Amazon Customer Review dataset
- perform minor cleaning of the relevant fields
- select reviews with 10 - 50 words
- group the records by `product_id`
- select products with both positive (5 stars) and negative (1 star) reviews
- select the first "most helpful" review for each product and add it to the dataset

In addition, the script builds a test set that consists of _conditional prompts_ for products with only positive or only negative reviews (i.e., products that are not in our training set), which will be used to assess the model generations in the [generate_sentiment_swapped_reviews.py](./generate_sentiment_swapped_reviews.py). 

Run the following command to create the `Video_Games_v1_00` data subset (note that the datasets with the default script parameters have already been created and are available in the [data](./data) directory): 

```bash
python create_dataset.py --data-subset Video_Games_v1_00
```
> Note: The full video game dataset has roughly 1.8 million rows (~1.2GB).

If you want to use the `Apparel_v1_00` data subset, you can run:

```bash
python create_dataset.py --data-subset Apparel_v1_00
```
> Note: The full apparel dataset is fairly large, with roughly 6 million rows (~2.1 GB). 

All the available command-line options can be viewed by running:

```bash
python create_dataset.py -h
```

### Fine tune with Gretel GPT

Fine tuning is carried out in the [fine_tune_with_gretel_gpt.py](./fine_tune_with_gretel_gpt.py) script, which configures a session with [Gretel Cloud](https://console.gretel.ai/) and submits the job using the conditional prompts that were created in the previous step.

Fine tune the model on the `Video_Games_v1_00` data subset by running:

```bash
python fine_tune_with_gretel_gpt.py --data-subset Video_Games_v1_00
```

The `Video_Games_v1_00` data subset contains ~16,000 review pairs. The fine-tuning process should take a few hours. You can check the job status in the [Gretel Cloud console](https://console.gretel.ai/projects). 

### Swap product review sentiments with Gretel GPT

We are finally ready to generate some product reviews! Follow along with the [gretel-gpt-sentiment-swap.ipynb](./gretel-gpt-sentiment-swap.ipynb) notebook to see how we prompt our fine-tuned model in the Gretel Cloud to swap the sentiment of product reviews. 

Alternatively, you can use the [generate_sentiment_swapped_reviews.py](./generate_sentiment_swapped_reviews.py) script to generate the reviews from the command line, which will write the review pairs in a text file in the `model-generations` directory for easy viewing.

Generate sentiment-swapped reviews for the `Video_Games_v1_00` data subset by running:

```bash
python generate_sentiment_swapped_reviews.py --data-subset Video_Games_v1_00
```
