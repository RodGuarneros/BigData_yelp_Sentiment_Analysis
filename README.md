# BigData_yelp_Sentiment_Analysis
A supervised ML model to make a setiment analysis with Yelp opinions
- Pipeline:
  - StringIndexer(inputCol='class',outputCol='label')
  - Tokenizer(inputCol="text", outputCol="token_text")
  - StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
  - HashingTF(inputCol="token_text", outputCol='hash_token')
  - IDF(inputCol='hash_token', outputCol='idf_token')
  - VectorAssembler(inputCols=['idf_token', 'length'], outputCol='features')

