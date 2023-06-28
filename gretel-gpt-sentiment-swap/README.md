<img src="../assets/gretel_icon.jpg" height="80" width="80" align="left" style="margin-right: 0px"/>

# Gretel GPT Sentiment Swap

This repo contains the code and notebooks for the Gretel GPT Sentiment Swap blog post.

## 😸 Use case overview

The code in this folder demonstrates how to use the [Gretel GPT API](https://docs.gretel.ai/reference/synthetics/models/gretel-gpt) to fine-tune and prompt a large language model to swap the sentiments of product reviews. Specifically, given a product review with a particular sentiment (positive or negative), our fine-tuned model will generate a new review with the opposite sentiment.

This use case is relevant for a variety of practical applications. For example, a company may want to train a review sentiment classifier for a group of products for which the vast majority of labeled examples are negative. In this case, the process we show here could be used to generate synthetic positive reviews to balance the dataset.

## 💾 Dataset for fine-tuning

We fine-tune Gretel GPT using subsets of the [Amazon Customer Reviews](https://huggingface.co/datasets/amazon_us_reviews). For us, the important columns of this dataset are `product_id`, `product_title`, `star_rating`, and `review_body`. We use the `star_rating` the as a proxy for sentiment, where 5 star is positive and 1 star is negative. By grouping the records by `product_id`, we build a dataset with both positive and negative reviews for a thousands of products. We then use this dataset to fine-tune Gretel GPT to generate positive and negative reviews for each product. 

## 🤖 Example Gretel GPT generations

We fine-tuned two Gretel GPT models on two different subsets of the Amazon Customer Reviews dataset: `Apparel_v1_001` and  `Video_Games_v1_00`. Here are some example generations for each (the highlighted text are the model generations):

### Apparel_v1_001

<div style="padding-left: 30px;">

---

Product Name: Little Boys Paw Patrol Toddler Hat<br>
Number of Stars: 5<br>
Review: Great hat and great fit...well made. Purchased for my nephew and he loves it..

Product Name: Little Boys Paw Patrol Toddler Hat<br>
Number of Stars: 1<br>
Review: <mark>The hat I received had a big rip in the back and no tags inside to put the child's name. It is also a different hat than what is pictured in the ad.</mark>

---

Product Name: Disney Girls' 2 Piece Minnie Knit Pant Set<br>
Number of Stars: 1<br>
Review: As with all the Minnie Mouse clothes my daughter got for Christmas, this shrunk from a 4t to not fitting my 2 year old daughter at all.  So disappointed with Disney clothing.

Product Name: Disney Girls' 2 Piece Minnie Knit Pant Set<br>
Number of Stars: 5<br>
Review: <mark>The product is as described, and as expected. It looks great on my daughter, and the material is of great quality. Very happy with the product.</mark>

</div>

### Video_Games_v1_00

<div style="padding-left: 30px;">

---

Product Name: Just Dance 4<br>
Number of Stars: 5<br>
Review: It is a really good game. My daughter enjoys playing and working out with it

Product Name: Just Dance 4<br>
Number of Stars: 1<br>
Review: <mark>This game is so old. It was made in 2012, and now that I bought it, they are making a new one that is way better. This one has a really low budget. I suggest not to buy it, get the newer one that is way better.</mark>

---

Product Name: Red Nunchuck Wired Controller Compatible with Nintendo Wii<br>
Number of Stars: 1<br>
Review: This product seemed like it would be great and a great price. It quit working after the second use. Complete junk.

Product Name: Red Nunchuck Wired Controller Compatible with Nintendo Wii<br>
Number of Stars: 5<br>
Review: <mark>This is an excellent wired nunchuck. I purchased it to use with my old Wii and I haven't had any problems with it. Works just as it should.</mark>

---
</div>

## 👩‍🔬 Steps to reproduce our results

