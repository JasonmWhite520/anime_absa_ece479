from urllib.request import urlopen
from bs4 import BeautifulSoup
import time

def reviews(url):
    reviews = []
    for page in range(1, 51):
        page = urlopen(url)
        html_bytes = page.read()
        html = html_bytes.decode("utf-8")
        soup = BeautifulSoup(html)
        reviews.extend(soup.find_all("div", attrs={"class": "review-element js-review-element"}))
        for review in reviews:
            print(review.prettify())
        time.sleep(0.5)

    clean_reviews = []
    for review in reviews:
        # TODO: Might need to filter by class before removing spans to handle short reviews.
        recommend = review.find("div", class_="tag").get_text()

        review_text = review.find("div", attrs={"class": "text"})

        if review_text.span is not None:
            review_text.span.extract()
            extended_review = review_text.span.extract().get_text()
            clean_reviews.append((recommend, review_text.get_text() + extended_review))
    for clean_review in clean_reviews:
        print(clean_review)
        print(
            "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")



