from pyspark.sql.functions import lower, regexp_replace, split, explode
from pyspark.ml.feature import NGram

# Read input text data and set wholetext=True to handle sentences spanning multiple lines
data = spark.read.text(path, wholetext=True)

# Convert text to lowercase and replace . with " $ "
data = data.withColumn("value", lower(regexp_replace("value", "\\.", " $ ")))

# Remove special characters
data = data.withColumn("value", regexp_replace("value", "[^a-zA-Z0-9\\s$]", ""))

# Tokenize the text based on space
data = data.withColumn("words", split("value", " "))

# Create a list of all sequences of 1 to k words
ngrams = NGram(n=range(1, k+1), inputCol="words", outputCol="ngrams")
data = ngrams.transform(data)

# Use flatMap() to create a key-value pair for each sequence and its subsequent word
pairs = data.select(explode("ngrams").alias("ngram"), "words") \
            .withColumn("subsequent", split("words", ",").getItem(k)) \
            .select("ngram", "subsequent")

# Use reduceByKey() to count the frequency of each sequence-subsequent word pair
counts = pairs.rdd.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y)
[19:31, 14/02/2023] Yashvardhan: from pyspark.sql.functions import lower, regexp_replace, split, explode
from pyspark.ml.feature import NGram

# Read input text data and set wholetext=True to handle sentences spanning multiple lines
data = spark.read.text(path, wholetext=True)

# Convert text to lowercase and replace . with " $ "
data = data.withColumn("value", lower(regexp_replace("value", "\\.", " $ ")))

# Remove special characters
data = data.withColumn("value", regexp_replace("value", "[^a-zA-Z0-9\\s$]", ""))

# Tokenize the text based on space
data = data.withColumn("words", split("value", " "))

# Create a list of all sequences of 1 to k words
ngrams = NGram(n=range(1, k+1), inputCol="words", outputCol="ngrams")
data = ngrams.transform(data)

# Use flatMap() to create a key-value pair for each sequence and its subsequent word
pairs = data.select(explode("ngrams").alias("ngram"), "words") \
            .withColumn("subsequent", split("words", ",").getItem(k)) \
            .select("ngram", "subsequent")

# Use reduceByKey() to count the frequency of each sequence-subsequent word pair
counts = pairs.rdd.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y)

# Use map() to group the subsequent words by their corresponding sequence
groups = counts.map(lambda x: (x[0][0], (x[0][1], x[1]))) \
                .groupByKey()

# Use mapValues() to calculate the probability of each subsequent word and sort them in descending order based on their probability
probs = groups.mapValues(lambda x: sorted(list(x), key=lambda y: y[1], reverse=True)[:n]) \
                .mapValues(lambda x: [(z[0], z[1]/sum(y[1] for y in x)) for z in x])

# Use map() to format the output as required and print the results
output = probs.flatMap(lambda x: [((w,), x[0], p) for w, p in x[1]]) \
                .map(lambda x: (", ".join(x[1]), " ".join(x[0]), round(x[2], 2))) \
                .collect()

for o in output:
    print(f"{o[0]}