<img src="../assets/gretel_icon.jpg" height="65" width="65" align="left" style="margin-right: -5px"/>

# Gretel GPT Sentiment Swap

This repo contains the code and notebooks for the Gretel GPT Sentiment Swap blog post.

## 😸 Use case overview

The code in this folder demonstrates how to use the [Gretel GPT API](https://docs.gretel.ai/reference/synthetics/models/gretel-gpt) to fine tune and prompt a large language model to swap the sentiments of product reviews. Specifically, given a product review with a particular sentiment (positive or negative), our fine-tuned model will generate a new review with the opposite sentiment.

This use case is relevant for a variety of practical applications. For example, a company may want to train a sentiment classification model on the comment section of their company blog, where negative comments are significantly underrepresented. In this case, similar steps to those we show here could be used to generate synthetic negative comments to balance the dataset.

## 💾 Dataset for fine-tuning

We fine tune Gretel GPT using subsets of the [Amazon Customer Reviews](https://huggingface.co/datasets/amazon_us_reviews). For us, the important columns of this dataset are `product_id`, `product_title`, `star_rating`, and `review_body`. We use the `star_rating` the as a proxy for sentiment, where 5 star is positive and 1 star is negative. By grouping the records by `product_id`, we build a dataset with both positive and negative reviews for a thousands of products. We then use this dataset to fine tune Gretel GPT to generate positive and negative reviews for each product. 

## 🤖 Example Gretel GPT generations

We fine tuned two Gretel GPT models on two different subsets of the Amazon Customer Reviews dataset: `Video_Games_v1_00` and  `Apparel_v1_001`. Here are some example generations for each (bold text indicates the model generations):

### Video_Games_v1_00

---

> Product Name: Just Dance 4<br>
> Number of Stars: 5<br>
> Review: It is a really good game. My daughter enjoys playing and working out with it
>
> Product Name: Just Dance 4<br>
> Number of Stars: 1<br>
> Review: **This game is so old. It was made in 2012, and now that I bought it, they are making a new one that is way better. This one has a really low budget. I suggest not to buy it, get the newer one that is way better.**

---

> Product Name: Red Nunchuck Wired Controller Compatible with Nintendo Wii<br>
> Number of Stars: 1<br>
> Review: This product seemed like it would be great and a great price. It quit working after the second use. Complete junk.
>
> Product Name: Red Nunchuck Wired Controller Compatible with Nintendo Wii<br>
> Number of Stars: 5<br>
> Review: **This is an excellent wired nunchuck. I purchased it to use with my old Wii and I haven't had any problems with it. Works just as it should.**

---

### Apparel_v1_001

---

> Product Name: Little Boys Paw Patrol Toddler Hat<br>
> Number of Stars: 5<br>
> Review: Great hat and great fit...well made. Purchased for my nephew and he loves it..
>
> Product Name: Little Boys Paw Patrol Toddler Hat<br>
> Number of Stars: 1<br>
> Review:  **The hat I received had a big rip in the back and no tags inside to put the child's name. It is also a different hat than what is pictured in the ad.**

---

> Product Name: Disney Girls' 2 Piece Minnie Knit Pant Set<br>
> Number of Stars: 1<br>
> Review: As with all the Minnie Mouse clothes my daughter got for Christmas, this shrunk from a 4t to not fitting my 2 year old daughter at all.  So disappointed with Disney clothing.
>
> Product Name: Disney Girls' 2 Piece Minnie Knit Pant Set<br>
> Number of Stars: 5<br>
> Review: **The product is as described, and as expected. It looks great on my daughter, and the material is of great quality. Very happy with the product.**

---

## 👩‍🔬 Steps to reproduce our results

### Installation

To run this example, the requirements in `requirements.txt`, which includes the [Gretel Python SDK](https://github.com/gretelai/gretel-python-client).

Preferably within a virtual environment, you can install the requirements using:

```bash
python -m pip -r requirements.txt
```
> Note: The results presented here were generated using Python 3.10.12.

Of course, you will also need to [install Jupyter](https://jupyter.org/install) to run the notebook.


### Create an account on the Gretel platform

If you haven't already, you will need to create an account on the [Gretel platform](https://console.gretel.ai/login). The free developer tier comes with 60 credits (5 hours of compute) per month, which is more than enough to fine tune one model on the `Video_Games_v1_00` data subset!

### Build the fine-tuning dataset

The [create_dataset.py](./create_dataset.py) script performs the following steps to create the fine-tuning dataset:
- downloads the given subset (passed as a command-line argument) of the Amazon Customer Review dataset
- performs minor cleaning of the relevant fields
- selects reviews with 10 - 50 words
- groups the records by `product_id`
- selects products with both positive (5 stars) and negative (1 star) reviews. 

In addition, the script builds a test set of products with only positive or only negative reviews, which will be used to assess the model generations in the [gretel-gpt-sentiment-swap.ipynb](./gretel-gpt-sentiment-swap.ipynb) notebook. 

Run the following command to create the `Video_Games_v1_00` data subset for fine-tuning (note that the datasets with the default script parameters have already been created and are available in the `data` directory): 

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

### Fine tune Gretel GPT

Fine tuning is carried out in the [fine_tune_with_gretel_gpt.py](./fine_tune_with_gretel_gpt.py) script, which configures a session with [Gretel Cloud](https://console.gretel.ai/) and submits a fine-tuning job using the conditional prompts that were created in the previous step.


```bash
python fine_tune_with_gretel_gpt.py --data-subset Video_Games_v1_00
```

For `Video_Games_v1_00`, there are ~16,000 5-star/1-star review pairs. You can check the job status in the [Gretel Cloud console](https://console.gretel.ai/projects). The fine-tuning process should take a few hours. 

### Swap product review sentiments with Gretel GPT

We are finally ready to generate some product reviews! Follow along with the [gretel-gpt-sentiment-swap.ipynb](./gretel-gpt-sentiment-swap.ipynb) notebook to see how we prompt our fine-tuned model in the Gretel Cloud to swap the sentiment of product reviews.
