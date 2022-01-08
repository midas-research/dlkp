"""Sample script for downloading dataset from using datasets from huggingface"""
from datasets import load_dataset

# get entire dataset
dataset = load_dataset("midas/kptimes", "raw")

# sample from the train split
print("Sample from training dataset split")
train_sample = dataset["train"][0]
print("Fields in the sample: ", [key for key in train_sample.keys()])
print("Tokenized Document: ", train_sample["document"])
print("Document BIO Tags: ", train_sample["doc_bio_tags"])
print("Extractive/present Keyphrases: ", train_sample["extractive_keyphrases"])
print("Abstractive/absent Keyphrases: ", train_sample["abstractive_keyphrases"])
print("\n-----------\n")

# sample from the validation split
print("Sample from validation dataset split")
validation_sample = dataset["validation"][0]
print("Fields in the sample: ", [key for key in validation_sample.keys()])
print("Tokenized Document: ", validation_sample["document"])
print("Document BIO Tags: ", validation_sample["doc_bio_tags"])
print("Extractive/present Keyphrases: ", validation_sample["extractive_keyphrases"])
print("Abstractive/absent Keyphrases: ", validation_sample["abstractive_keyphrases"])
print("\n-----------\n")

# sample from the test split
print("Sample from test dataset split")
test_sample = dataset["test"][0]
print("Fields in the sample: ", [key for key in test_sample.keys()])
print("Tokenized Document: ", test_sample["document"])
print("Document BIO Tags: ", test_sample["doc_bio_tags"])
print("Extractive/present Keyphrases: ", test_sample["extractive_keyphrases"])
print("Abstractive/absent Keyphrases: ", test_sample["abstractive_keyphrases"])
print("\n-----------\n")

# get the dataset only for keyphrase extraction
dataset = load_dataset("midas/kptimes", "extraction")

print("Samples for Keyphrase Extraction")

# sample from the train split
print("Sample from training dataset split")
train_sample = dataset["train"][0]
print("Fields in the sample: ", [key for key in train_sample.keys()])
print("Tokenized Document: ", train_sample["document"])
print("Document BIO Tags: ", train_sample["doc_bio_tags"])
print("\n-----------\n")

# sample from the validation split
print("Sample from validation dataset split")
validation_sample = dataset["validation"][0]
print("Fields in the sample: ", [key for key in validation_sample.keys()])
print("Tokenized Document: ", validation_sample["document"])
print("Document BIO Tags: ", validation_sample["doc_bio_tags"])
print("\n-----------\n")

# sample from the test split
print("Sample from test dataset split")
test_sample = dataset["test"][0]
print("Fields in the sample: ", [key for key in test_sample.keys()])
print("Tokenized Document: ", test_sample["document"])
print("Document BIO Tags: ", test_sample["doc_bio_tags"])
print("\n-----------\n")

# get the dataset only for keyphrase generation
dataset = load_dataset("midas/kptimes", "generation")

print("Samples for Keyphrase Generation")

# sample from the train split
print("Sample from training dataset split")
train_sample = dataset["train"][0]
print("Fields in the sample: ", [key for key in train_sample.keys()])
print("Tokenized Document: ", train_sample["document"])
print("Extractive/present Keyphrases: ", train_sample["extractive_keyphrases"])
print("Abstractive/absent Keyphrases: ", train_sample["abstractive_keyphrases"])
print("\n-----------\n")

# sample from the validation split
print("Sample from validation dataset split")
validation_sample = dataset["validation"][0]
print("Fields in the sample: ", [key for key in validation_sample.keys()])
print("Tokenized Document: ", validation_sample["document"])
print("Extractive/present Keyphrases: ", validation_sample["extractive_keyphrases"])
print("Abstractive/absent Keyphrases: ", validation_sample["abstractive_keyphrases"])
print("\n-----------\n")

# sample from the test split
print("Sample from test dataset split")
test_sample = dataset["test"][0]
print("Fields in the sample: ", [key for key in test_sample.keys()])
print("Tokenized Document: ", test_sample["document"])
print("Extractive/present Keyphrases: ", test_sample["extractive_keyphrases"])
print("Abstractive/absent Keyphrases: ", test_sample["abstractive_keyphrases"])
print("\n-----------\n")
