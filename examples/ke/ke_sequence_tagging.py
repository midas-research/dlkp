import requests
import time
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

from transformers import BertTokenizer
from transformers import BertForTokenClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

# We'll need the BertTokenizer for doing sequence tagging with Bert
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print("There are %d GPU(s) available." % torch.cuda.device_count())

        print("We will use the GPU:", torch.cuda.get_device_name(0))

    # If not...
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    return device


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_data():

    TRAIN_URL = "https://raw.githubusercontent.com/midas-research/keyphrase-extraction-as-sequence-labeling-data/master/SemEval-2017/train.txt"
    DEV_URL = "https://raw.githubusercontent.com/midas-research/keyphrase-extraction-as-sequence-labeling-data/master/SemEval-2017/dev.txt"
    TEST_URL = "https://raw.githubusercontent.com/midas-research/keyphrase-extraction-as-sequence-labeling-data/master/SemEval-2017/test.txt"

    response = requests.get(TRAIN_URL)
    _ = open("train.txt", "w").write(response.text)

    response = requests.get(DEV_URL)
    _ = open("valid.txt", "w").write(response.text)

    response = requests.get(TEST_URL)
    _ = open("test.txt", "w").write(response.text)


def parse_conll(filepath):
    # List of all sentences in the dataset.
    sentences = []
    labels = []

    # Lists to store the current sentence.
    tokens = []
    token_labels = []

    # Gather the set of unique labels.
    unique_labels = set()

    # Read the dataset line by line. Each line of the file
    # is either empty or has two tokens, separated by a tab.
    with open(filepath) as fp:

        # For each line in the file...
        for line in fp:
            # split the lines on the space character.
            line = line.rstrip().split()

            # If we encounter a blank line, it means we've completed the previous sentence.
            if not line:

                # Add the completed sentence.
                assert len(tokens) == len(token_labels)
                sentences.append(tokens)
                labels.append(token_labels)

                # Start a new sentence.
                tokens = []
                token_labels = []

            else:
                # Add the token and its label to the current sentence.
                tokens.append(line[0])
                token_labels.append(line[1])

                # Add the label to the set (no effect if it already exists).
                unique_labels.add(line[1])

    return sentences, labels, unique_labels


def prepare_tokenized_input(sentences):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # Reconstruct the sentence--otherwise `tokenizer` will interpret the list
        # of string tokens as having already been tokenized by BERT.
        sent_str = " ".join(sent)

        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent_str,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=50,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict["input_ids"][0])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict["attention_mask"][0])

    # Print sentence 0, now as a list of IDs.
    print("Original: ", sentences[0])
    print("Token IDs:", input_ids[0])
    print("Masks:", attention_masks[0])

    return input_ids, attention_masks


def map_labels(unique_labels):
    # Map each unique label to an integer.
    label_map = {}

    # For each label...
    for (i, label) in enumerate(unique_labels):
        # Map it to its integer.
        label_map[label] = i

    print(label_map)

    return label_map


def add_null_labels(input_ids, labels, label_map):
    # New labels for all of the input sentences.
    new_labels = []

    # The special label ID we'll give to "extra" tokens.
    null_label_id = -100

    # For each sentence...
    for (sen, orig_labels) in zip(input_ids, labels):

        # Create a new list to hold the adjusted labels for this sentence.
        padded_labels = []

        # This will be our index into the original label list.
        orig_labels_i = 0

        # For each token in the padded sentence...
        for token_id in sen:

            # Pull the value out of the tensor.
            token_id = token_id.numpy().item()

            # If `[PAD]`, `[CLS]`, or `[SEP]`...
            if (
                (token_id == tokenizer.pad_token_id)
                or (token_id == tokenizer.cls_token_id)
                or (token_id == tokenizer.sep_token_id)
            ):

                # Assign it the null label.
                padded_labels.append(null_label_id)

            # If the token string starts with "##"...
            elif tokenizer.ids_to_tokens[token_id][0:2] == "##":

                # It's a subword token, and not part of the original dataset, so
                # assign it the null label.
                padded_labels.append(null_label_id)

            # If it's not any of the above...
            else:

                # This token corresponds to one of the original ones, so assign it
                # it's original label.

                # Look up the label for this token.
                label_str = orig_labels[orig_labels_i]

                # Map the label to its ID, and assign it.
                padded_labels.append(label_map[label_str])

                # Increment our index into the original labels list.
                orig_labels_i += 1

        # If we did this right, then the new `padded_labels` list should match
        # the length of the tokenized sentence.
        assert len(sen) == len(padded_labels)

        # Store the updated labels list for this sentence.
        new_labels.append(padded_labels)

    print("Final labels: ", new_labels[0])

    return new_labels


def convert_to_tensors(input_ids, attention_masks, labels):
    # Convert the lists into PyTorch tensors.

    # `input_ids` is a list of tensor arrays--stack them into a matrix with size
    # [7,660  x  50].
    pt_input_ids = torch.stack(input_ids, dim=0)

    # `attention_masks` is a list of tensor arrays--stack them into a matrix with
    # size [7,660  x  50].
    pt_attention_masks = torch.stack(attention_masks, dim=0)

    # Labels is a list of lists. Convert it into a tensor matrix with size
    # [7,660  x  50].
    pt_labels = torch.tensor(labels, dtype=torch.long)

    return pt_input_ids, pt_attention_masks, pt_labels


def train_model(train_data, valid_data, device):
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training dataset...
        for step, batch in enumerate(train_data):
            # Progress update every 5 batches.
            if step % 5 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                        step, len(train_data), elapsed
                    )
                )

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # In PyTorch, calling `model` will in turn call the model's `forward`
            # function and pass down the arguments. The `forward` function is
            # documented here:
            # https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.TokenClassifierOutput
            result = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            loss = result.loss

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training dataset.
        avg_train_loss = total_loss / len(train_data)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")

    # Use plot styling from seaborn.
    sns.set(style="darkgrid")

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(loss_values, "b-o")

    # Label the plot.
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()


if __name__ == "__main__":
    # get the dataset
    get_data()

    # process the training dataset
    # parse the conll format
    train_sentences, train_labels, unique_labels = parse_conll("train.txt")
    label_mapping = map_labels(unique_labels)
    # convert the tokens to their unique ids and also add attention masks
    train_token_ids, train_attention_masks = prepare_tokenized_input(train_sentences)
    # add the null labels for special tokens like [SEP], [CLS], etc
    final_train_labels = add_null_labels(train_token_ids, train_labels, label_mapping)
    # convert the processed dataset to tensors
    pt_train_token_ids, pt_train_attention_masks, pt_train_labels = convert_to_tensors(
        train_token_ids, train_attention_masks, final_train_labels
    )

    # process the validation dataset
    # parse the conll format
    valid_sentences, valid_labels, _ = parse_conll("valid.txt")
    # convert the tokens to their unique ids and also add attention masks
    valid_token_ids, valid_attention_masks = prepare_tokenized_input(valid_sentences)
    # add the null labels for special tokens like [SEP], [CLS], etc
    final_valid_labels = add_null_labels(valid_token_ids, valid_labels, label_mapping)
    # convert the processed dataset to tensors
    pt_valid_token_ids, pt_valid_attention_masks, pt_valid_labels = convert_to_tensors(
        valid_token_ids, valid_attention_masks, final_valid_labels
    )
    # process the test dataset
    # parse the conll format
    test_sentences, test_labels, _ = parse_conll("test.txt")
    # convert the tokens to their unique ids and also add attention masks
    test_token_ids, test_attention_masks = prepare_tokenized_input(test_sentences)
    # add the null labels for special tokens like [SEP], [CLS], etc
    final_test_labels = add_null_labels(test_token_ids, test_labels, label_mapping)
    # convert the processed dataset to tensors
    pt_test_token_ids, pt_test_attention_masks, pt_test_labels = convert_to_tensors(
        test_token_ids, test_attention_masks, final_test_labels
    )

    # Convert the training inputs into a TensorDataset.
    train_dataset = TensorDataset(
        pt_train_token_ids, pt_train_attention_masks, pt_train_labels
    )
    # Convert the validation inputs into a TensorDataset.
    valid_dataset = TensorDataset(
        pt_valid_token_ids, pt_valid_attention_masks, pt_valid_labels
    )
    # Convert the test inputs into a TensorDataset.
    test_dataset = TensorDataset(
        pt_test_token_ids, pt_test_attention_masks, pt_test_labels
    )

    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = 32

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        valid_dataset,  # The validation samples.
        sampler=SequentialSampler(valid_dataset),  # Pull out batches sequentially.
        batch_size=batch_size,  # Evaluate with this batch size.
    )

    # Load BertForTokenClassification
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=len(label_mapping)
        + 1,  # The number of output labels--18 for our NER dataset
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    # Load the AdamW optimizer
    optimizer = AdamW(
        model.parameters(), lr=5e-5, eps=1e-8  # args.learning_rate  # args.adam_epsilon
    )

    # Number of training epochs
    epochs = 4

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    device = get_device()

    train_model(
        train_data=train_dataloader, valid_data=validation_dataloader, device=device
    )

    # Prediction on test set
    # Set the batch size.
    batch_size = 32

    # Create the DataLoader.
    prediction_data = TensorDataset(
        pt_test_token_ids, pt_test_attention_masks, pt_test_labels
    )
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=batch_size
    )

    print("Predicting labels for {:,} test sentences...".format(len(pt_test_token_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            result = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                return_dict=True,
            )

        logits = result.logits

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print("    DONE.")

    # First, combine the results across the batches.
    all_predictions = np.concatenate(predictions, axis=0)
    all_true_labels = np.concatenate(true_labels, axis=0)

    print("After flattening the batches, the predictions have shape:")
    print("    ", all_predictions.shape)

    # Next, let's remove the third dimension (axis 2), which has the scores
    # for all 18 labels.

    # For each token, pick the label with the highest score.
    predicted_label_ids = np.argmax(all_predictions, axis=2)

    print("\nAfter choosing the highest scoring label for each token:")
    print("    ", predicted_label_ids.shape)

    # Finally, for the sake of scoring, we don't actually care about the different
    # sentences--we just look at whether the model made correct predictions for the
    # individual tokens.

    # Eliminate axis 0, which corresponds to the sentences.
    predicted_label_ids = np.concatenate(predicted_label_ids, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    print("\nAfter flattening the sentences, we have predictions:")
    print("    ", predicted_label_ids.shape)
    print("and ground truth:")
    print("    ", all_true_labels.shape)

    # Construct new lists of predictions which don't include any null tokens.
    real_token_predictions = []
    real_token_labels = []

    # For each of the input tokens in the dataset...
    for i in range(len(all_true_labels)):

        # If it's not a token with a null label...
        if not all_true_labels[i] == -100:
            # Add the prediction and the ground truth to their lists.
            real_token_predictions.append(predicted_label_ids[i])
            real_token_labels.append(all_true_labels[i])

    print(
        "Before filtering out `null` tokens, length = {:,}".format(len(all_true_labels))
    )
    print(
        " After filtering out `null` tokens, length = {:,}".format(
            len(real_token_labels)
        )
    )

    # Calculate the F1 score. Because this is a multi-class problem, we have
    # to set the `average` parameter. TODO - What does `micro` do?
    f1 = f1_score(real_token_labels, real_token_predictions, average="micro")

    print("F1 score: {:.2%}".format(f1))
