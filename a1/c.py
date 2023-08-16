from pyspark.sql.functions import lower, regexp_replace, split, explode, concat_ws, collect_list
from pyspark.ml.feature import NGram
from pyspark.sql import Window

# set parameters
k = 3
n = 5

# read input data
input_df = spark.read.text('path/to/input/data', wholetext=True)

# preprocess text
preprocessed_df = input_df.withColumn('text', lower(regexp_replace('value', r'[^a-z0-9\s]', ''))) \
    .withColumn('text', regexp_replace('text', r'\.', ' $ ')) \
    .withColumn('words', split('text', '\s+'))

# create k-grams
ngram = NGram(n=k, inputCol='words', outputCol='kgrams')
kgrams_df = ngram.transform(preprocessed_df)

# extract previous words and next word
extract_df = kgrams_df.withColumn('kgram', explode('kgrams')) \
    .withColumn('previous_words', concat_ws(' ', *[f'kgram[{i}]' for i in range(k-1)])) \
    .withColumn('next_word', f'kgram[{k-1}]')

# count occurrences of each (previous words, next word) tuple
count_df = extract_df.groupBy('previous_words', 'next_word').count()

# calculate total count for each unique previous words
total_df = count_df.groupBy('previous_words').agg({'count': 'sum'}) \
    .withColumnRenamed('sum(count)', 'total_count')

# join count_df with total_df and calculate probability
join_df = count_df.join(total_df, 'previous_words')
prob_df = join_df.withColumn('probability', f'count / total_count')

# rank next words by probability for each unique previous words
w = Window.partitionBy('previous_words').orderBy(prob_df['probability'].desc())
ranked_df = prob_df.withColumn('rank', f'row_number().over({w})').filter(f'rank <= {n}')

# format output as string
output_df = ranked_df.withColumn('output', concat_ws(' ', 'next_word', 'probability')) \
    .groupBy('previous_words').agg(collect_list('output').alias('outputs')) \
    .withColumn('outputs', concat_ws('\n', 'outputs'))

# save output to text file
output_df.write.text('path/to/output', compression=None, sep='\n')