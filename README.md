# Disco: Imageset Diversity Score
Evaluating diversity of image search results using computer vision

## Problem

Image search engines suffer from human stereotypes associating inherent backgrounds (e.g. age, race, gender) with social status (e.g. occupations). For example, when a user searches “professor” on a popular search engine, 14 out of 15 image search results are old white males (July 2019). Not only does lack of diversity in image search results impact the user’s perspective of the world (thereby contributing to the vicious cycle of human stereotypes and search results reinforcing each other), it also negatively impacts user satisfaction of the search results (source: Microsoft Bing Internal Research).

Search engines like Microsoft Bing are pouring resources into increasing diversity in image search results. However, measuring the effect of the changes introduced to the search algorithms remains a difficult challenge. In status quo, human judges are evaluating the image search results manually. This evaluation method is expensive, slow, and prone to human error. To scale the image search diversity improvement initiatives, the world needs a scalable and automated method of measuring image search results.

## Solution

[Disco](https://github.com/varsanc/disco) automates the process of evaluating diversity of image datasets. In order to compute a numeric score which is positively correlated with the diversity represented in the images of an image dataset, Disco does the following for each image dataset:
1. Extract faces in each image (Disco draws bounding boxes around faces).
2. Generate face embedding vectors for each face.  
3. Calculate the average pairwise distance among face embedding vectors.

Note: Disco supports detecting multiple faces in a single image. An image dataset with 6 images in which first 5 images each contain a single person with trait X and 1 image contains 5 people with trait Y, Disco will produce a high diversity score.

## Usage

Anyone can import Disco into their own application as a Python class to evaluate their own image datasets.  

[A Jupyter notebook](https://github.com/varsanc/disco/blob/master/Imageset%20Diversity%20Score.ipynb) is also included as an example.

Disco supports two input file types:
1. (Microsoft Internal Use): TSV output file from Avatar. Avatar is an internal tool at Microsoft for searching, scraping, and judging search results from different search engines given text queries.
2. JSON file. Each line is separated by a line break “\n”. Each line contains a JSON object with the following schema: {“image_uri”: URL pointing to an image, “imageset”: String describing the imageset to which the given image belongs}.  

Disco output can be interpreted either from an output file (JSON file. Each line is separated by a line break “\n”. Each line contains a JSON object with the following schema: {“imageset”: String, “diversity_score”: Number}), or from the terminal using the print_imageset_diversity_score() function.

Deploy a [FaceNet](https://github.com/varsanc/disco/tree/master/facenet) server and set your server address as a constant variable in Disco.

## Experiment

We hypothesized that given a pair of imagesets from which human judges perceive a disparity in diversity, Disco will compute a higher diversity score for the imageset that human judges consider to have a higher diversity. To test our hypothesis, we conducted the following experiment:
1. We collected 20 pairs of imagesets labeled with clear winner in diversity perceived by human judges. Microsoft Bing is testing a new algorithm for ensuring diversity in images displayed as search results. We selected 20 search queries for which the new algorithm returns images with more diversity than does the old algorithm according to human judges. All 20 search queries are titles of occupations. For each of the 20 search queries, we collected an imageset of search results returned by the old algorithm and an imageset of search results returned by the new algorithm. Our hypothesis would be valid if Disco returns a higher diversity score for the imagesets of search results returned by the new algorithm.
2. We calculated and compared the diversity scores between each pair of imagesets.
3. Disco correctly calculated a higher diversity score from imagesets with more diversity in 15 out of 20 imagesets, and calculated the same diversity score for both imagesets in the remaining 5 imagesets.

| Search Query | Less Diverse Imageset | More Diverse Imageset |
| ------------ | --------------------- | --------------------- |
| Banker       | 36.16                 | 37.88                 |
| CEO          | 33.22                 | 35.21                 |
| Comedian     | 27.74                 | 35.08                 |
| Janitor      | 23.73                 | 33.00                 |
| Jockey       | 25.76                 | 27.18                 |
| Journalist   | 38.61                 | 38.00                 |
| Librarian    | 32.96                 | 32.93                 |
| Lumberjack   | 25.07                 | 31.85                 |
| Manager      | 37.48                 | 38.07                 |
| Nanny        | 30.87                 | 36.06                 |
| Neurologist  | 37.88                 | 37.10                 |
| Nurse        | 34.99                 | 36.24                 |
| Optician     | 34.89                 | 34.76                 |
| Optometrist  | 35.71                 | 36.85                 |
| Organist     | 28.81                 | 36.98                 |
| Painter      | 37.98                 | 39.50                 |
|Paleontologist| 31.71                 | 32.62                 |
| Plumber      | 30.54                 | 33.82                 |
| Poet         | 27.86                 | 27.84                 |
| Professor    | 34.30                 | 35.89                 |

This result means Disco can correctly identify the more diverse imageset in most cases, and defer the unsure decisions to human judges. Since Disco does not produce a higher diversity score for imagesets with less diversity, companies can use Disco to determine the winners for most comparison pairs, and only fall back on human judges for the remaining uncertain comparison pairs.

From this, we conclude that Disco is a useful tool for evaluating the impact of search algorithm changes on the diversity of image search results.  
