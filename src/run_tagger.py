from dlkp.extraction.tagger import KeyphraseTagger

tagger = KeyphraseTagger.load(model_name_or_path="")
kps = tagger.predict(
    "Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned"
)

print(kps)
