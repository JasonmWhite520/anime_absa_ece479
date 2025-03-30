from anime_absa import aspect_term_extraction, aspect_term_sentiment_classification, get_overall_review_sentiment

blah_review = """iamjoe
    Recommended
    Death Note is original, awesome, and a great anime to watch. The only reason it isn't perfect is because of Near who has to be the lamest enemy ever. Also, the ending was a total cop-out. DUMB, DUMB, DUMB.
    
    The art wasn't that great, but I sure did like this anime anyway.
    
    The sound was fitting; it suited the tense atmosphere and scenes.
    
    The characters were all intriguing and deep, especially Light. It's cool to see such an evil protagonist in an anime. He's certainly original.
    
    If you want a super-cool anime to watch, watch Death Note. Except for L and Near, who made everything totally unfabulous.
    Reviewer’s Rating: 9"""
# print(aspect_term_extraction("The movie was horrible, it was they like they excelled in kicking puppies and kittens through film"))
#
# print(aspect_term_extraction("it was they like they excelled in kicking puppies and kittens through film"))
#
# print(aspect_term_extraction("The movie was they like they excelled in kicking puppies and kittens through film"))
#
# print(aspect_term_extraction("The movie was they like they excelled in making it terrible as if they were kicking puppies and kittens through film"))
#
# print(aspect_term_extraction("The movie was horrible"))
#
# print(aspect_term_extraction("My waifu Tenma was great as a star, it's nice when she yells at people"))
#
# print(aspect_term_extraction("Who let the dogs out"))
#
# print(aspect_term_extraction("The music was great, it had a light melody that really touched the senses in the most intimate ways that is meaningful"))
#
# print(aspect_term_extraction("The muvie had an all cast teim that really knucked the ball out of the park. I really enjoyed the the plut and soundtrek."))

# print(aspect_term_sentiment_classification("The movie was bad","movie"))
#
# print(aspect_term_sentiment_classification("The movie was good","movie"))
#
# print(aspect_term_sentiment_classification("The movie was ok","movie"))
#
# print(aspect_term_sentiment_classification("The movie was not good","movie"))
#
# print(aspect_term_sentiment_classification("The movie was not ok","movie"))
#
# print(aspect_term_sentiment_classification("The movie was pleasant","movie"))
#
# print(aspect_term_sentiment_classification("The movie was pleasent","movie"))
#
# print(aspect_term_sentiment_classification("The movie was bid","movie"))
#
# print(aspect_term_sentiment_classification("The movie was goud","movie"))

print(get_overall_review_sentiment("The art was good, the music good, the plot was bad"))