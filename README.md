# BigData
## Yelp Sentiment Analysis
## By Rodrigo Guarneros Gutiérrez

Sentiment analysis: Built a supervised ML model using PySpark, Google Colab, and AWS for sentiment analysis of Yelp opinions. 

A supervised ML model to make a setiment analysis with Yelp opinions
- Pipeline:
  - StringIndexer(inputCol='class',outputCol='label')
  - Tokenizer(inputCol="text", outputCol="token_text")
  - StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
  - HashingTF(inputCol="token_text", outputCol='hash_token')
  - IDF(inputCol='hash_token', outputCol='idf_token')
  - VectorAssembler(inputCols=['idf_token', 'length'], outputCol='features')

