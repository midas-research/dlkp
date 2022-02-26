from dlkp import extract_from_text

result = extract_from_text(
    text_list=[
        "Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned"
    ],
    model_name_or_path="/media/nas_mount/Debanjan/amardeep/dlkp_out/inpec_crf_debug/",
    use_CRF=True,
)

print(result)
