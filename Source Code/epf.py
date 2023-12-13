from module import *
enc_type = "aap"

# Project/dataset directory path
main_path = "C:/Users/Jerry/Desktop/Implementation/"
dataset_bacteria = "bacteria/bacillus"
dataset_parasite = "parasite/plasmodium"
dataset_virus = "virus/human_sars_cov2"
dataset_plant = "plant/Ara_PsyGorHpa"

project_type = "project/epf"

project_type_with_underscore = str(project_type.split("/")[0]) + "_" + str(project_type.split("/")[1])
file_tag = str(time.time()).split(".")[0]

# # create temporary directory for storing files if it does not exist
# tmp_dir = createDir(main_path+project_type+"/"+"tmp/")
#
# ===================Data Preprocessing starts here =============================
print("Preprocessing Starts...")
dataset_ls = [dataset_bacteria, dataset_parasite, dataset_virus, dataset_plant]
hpi_data = pd.DataFrame()
start_group_id = 1
all_group_ids = []
for dataset in dataset_ls:
    # read dataset files (fasta format) from the project directory into list
    int_host = readFasta(main_path+dataset+"/"+"inthost.fasta")
    int_pathogen = readFasta(main_path+dataset+"/"+"intpathogen.fasta")
    nonint_host = readFasta(main_path+dataset+"/"+"noninthost.fasta")
    nonint_pathogen = readFasta(main_path+dataset+"/"+"nonintpathogen.fasta")

    # delete any invalid protein sequence
    print("Removing invalid protein pairs...")
    int_host, int_pathogen = deleteInvalidProt(int_host, int_pathogen, "interacting")
    nonint_host, nonint_pathogen = deleteInvalidProt(nonint_host, nonint_pathogen, "noninteracting")

    print(f"int_host: {len(int_host)}")
    print(f"int_pathogen: {len(int_pathogen)}")
    print(f"nonint_host: {len(nonint_host)}")
    print(f"nonint_pathogen: {len(nonint_pathogen)}")

    # Group protein sequences for each data and create group ids
    grouped_int_host, int_host_group_ids = groupProteinSequences(int_host, start_group_id)
    grouped_nonint_host, nonint_host_group_ids = groupProteinSequences(nonint_host, max(int_host_group_ids, default=None)+1)
    max_group_id = max(nonint_host_group_ids, default=None)
    start_group_id = max_group_id + 1
    int_nonint_group_ids = int_host_group_ids + nonint_host_group_ids
    all_group_ids = all_group_ids + int_nonint_group_ids

    # ==============Computing MPF=======================================
    concat_int = [[int_host[i][0] + "_" + int_pathogen[i][0], int_host[i][1] + ""
                   + int_pathogen[i][1]] for i in range(len(int_host))]

    concat_nonint = [[nonint_host[i][0] + "_" + nonint_pathogen[i][0], nonint_host[i][1] + ""
                      + nonint_pathogen[i][1]] for i in range(len(nonint_host))]

    # set output file path for the feature vectors
    int_path = tmp_dir + "int_" + file_tag
    nonint_path = tmp_dir + "nonint_" + file_tag

    # perform the feature encoding technique (Note: This operations takes time)
    print("Performing mpf protein sequence feature encoding...")
    intx_path = generateFeatures(convertToFasta(concat_int, int_path + ".fasta"), int_path + "_x.csv")
    nonintx_path = generateFeatures(convertToFasta(concat_nonint, nonint_path + ".fasta"), nonint_path + "_x.csv")

    # Remove index column from files
    intx = pd.read_csv(intx_path, delimiter='\t', encoding='latin-1').iloc[:, 1:]
    nonintx = pd.read_csv(nonintx_path, delimiter='\t', encoding='latin-1').iloc[:, 1:]
    # ================ End Computing MPF ===============================

    # =================Computing IPF ===================================
    # set output file path for the feature vectors
    int_host_path = tmp_dir + "int_host_" + file_tag
    int_pathogen_path = tmp_dir + "int_pathogen_" + file_tag
    nonint_host_path = tmp_dir + "nonint_host_" + file_tag
    nonint_pathogen_path = tmp_dir + "nonint_pathogen_" + file_tag

    # perform the feature encoding technique (Note: This operations takes time)
    print("Performing ipf protein sequence feature encoding...")
    int_hostx_path = generateFeatures(
        convertToFasta(int_host, int_host_path + ".fasta"), int_host_path + "_x.csv")
    int_pathogenx_path = generateFeatures(
        convertToFasta(int_pathogen, int_pathogen_path + ".fasta"), int_pathogen_path + "_x.csv")
    nonint_hostx_path = generateFeatures(
        convertToFasta(nonint_host, nonint_host_path + ".fasta"), nonint_host_path + "_x.csv")
    nonint_pathogenx_path = generateFeatures(
        convertToFasta(nonint_pathogen, nonint_pathogen_path + ".fasta"), nonint_pathogen_path + "_x.csv")

    # Remove index column from files
    int_hostx = pd.read_csv(int_hostx_path, delimiter='\t', encoding='latin-1').iloc[:, 1:]
    int_pathogenx = pd.read_csv(int_pathogenx_path, delimiter='\t', encoding='latin-1').iloc[:, 1:]
    nonint_hostx = pd.read_csv(nonint_hostx_path, delimiter='\t', encoding='latin-1').iloc[:, 1:]
    nonint_pathogenx = pd.read_csv(nonint_pathogenx_path, delimiter='\t', encoding='latin-1').iloc[:, 1:]
    # =================End Computing IPF ===============================

    # =================Computing EPF ===================================
    # combine Independent and Merged protein features and add label feature
    int_hp = pd.concat([int_hostx, int_pathogenx, intx], axis=1)
    nonint_hp = pd.concat([nonint_hostx, nonint_pathogenx, nonintx], axis=1)
    int_hp['label'] = 1
    nonint_hp['label'] = 0

    # # do data balancing
    # if len(int_hp) >= len(nonint_hp): int_hp = int_hp.sample(len(nonint_hp))
    # else: nonint_hp = nonint_hp.sample(len(int_hp))

    # # Sample 10%, 25%, 50% of the original data
    # enc_type = "aac"
    # percent = 10
    # percent_frac = percent / 100
    # int_hp = int_hp.sample(frac=percent_frac, random_state=seed)
    # nonint_hp = nonint_hp.sample(frac=percent_frac, random_state=seed)

    # combine interacting and non interacting protein
    hpi_int_nonint = pd.concat([int_hp, nonint_hp], axis=0, ignore_index=True)
    print(hpi_int_nonint.shape)
    print(hpi_int_nonint['label'].value_counts())

    # merge previous data with the existing data i.e all organism datasets
    hpi_data = pd.concat([hpi_data, hpi_int_nonint], axis=0, ignore_index=True)
all_group_ids = pd.DataFrame(all_group_ids, columns=['group_id'])
# ===================end of data preprocessing ==============================

# ==================Feature Extraction=========================================
print(f"Performing feature extraction...")
# Splitting data into features (Xs) and label(y)
X_hpi = hpi_data.drop(columns='label')
y_hpi = hpi_data.label
X_hpi = np.nan_to_num(X_hpi) # replace NAN. INFINITY to Zero

# Feature selection with information gain =  same (AAC)
importances = mutual_info_classif(X_hpi, y_hpi)

# Selecting important features
print(f"Selecting important features...")
np.random.seed(seed)
feature_names = [f"feature {i}" for i in range(1, X_hpi.shape[1]+1)]
# feature_importances = pd.Series(importances, index=feature_names)
feature_importances = pd.Series(importances, index=feature_names).loc[lambda x : x != 0.00]
feature_importances_ls = feature_importances.index.tolist() # get important features as a list
X_hpi = pd.DataFrame(X_hpi, columns=feature_names) # rename original data features
X_hpi_fimp = X_hpi.loc[:, feature_importances_ls] # extract dataset based on important feature columns
print(X_hpi_fimp.shape)
# hpi_data = pd.concat([X_hpi_fimp, y_hpi], axis=1)

# Feature selection using correlation matrix
X_hpi_mc = multicollinearity(X_hpi_fimp)
print(X_hpi_mc.shape)
hpi_data = pd.concat([X_hpi_mc, y_hpi], axis=1)
print(hpi_data.shape)

# save extracted features to csv
preprocessedPath = createDir(main_path+project_type+"/"+ "preprocessed/")

hpi_data.to_csv(preprocessedPath + project_type_with_underscore + ".csv")
all_group_ids.to_csv(preprocessedPath + "groupids.csv")

print("Preprocessing Completed!")

print(hpi_data.shape)
print(hpi_data['label'].value_counts())

print("Feature extraction completed!")
# ==================End of Feature Extraction=========================================

# ===================Computational model starts ===============================
hpi_data = pd.read_csv(main_path+project_type+"/"+"preprocessed/" + project_type_with_underscore + ".csv", index_col=[0])
all_group_ids = pd.read_csv(main_path+project_type+"/"+"preprocessed/" + "groupids.csv", index_col=[0])
hpi_data = pd.concat([hpi_data, all_group_ids], axis=1)

print("\n Model Training Starts...")
# count number of interactions and non-interactions
print(hpi_data['label'].value_counts())
print(hpi_data.shape)
print(all_group_ids.value_counts())

# Output path directory
outputPath = createDir(main_path+project_type+"/"+ "output/")

# Create output file in excel
workbook = xlsxwriter.Workbook(outputPath + project_type_with_underscore + file_tag + ".xlsx")
worksheet = workbook.add_worksheet()

# Run experiment for both balanced and imbalanced dataset i.e range(0,2).
# If only balance, set range(0,1), if only imbalance, set range (1,2)
j = 0
for count in range(1, 2):
    if count == 0: analysis_type = "balanced"
    else: analysis_type = "imbalanced"
    fileTitle = str(analysis_type) + "_dataset_" + str(project_type_with_underscore)

    if analysis_type == "balanced":
        # separate interacting proteins from non-interacting ones
        int_hpi_data = hpi_data[hpi_data['label'] == 1]
        nonint_hpi_data = hpi_data[hpi_data['label'] == 0]

        # do data balancing
        if len(int_hpi_data) >= len(nonint_hpi_data): int_hpi_data = int_hpi_data.sample(len(nonint_hpi_data))
        else: nonint_hpi_data = nonint_hpi_data.sample(len(int_hpi_data))
        bal_hpi_data = pd.concat([int_hpi_data, nonint_hpi_data], axis=0, ignore_index=True)

        # separate balanced hpi features from labels
        X_hpi = bal_hpi_data.drop(columns='label')
        y_hpi = bal_hpi_data.label
        print("\nBalanced dataset label count")
        print(y_hpi.value_counts())
    else:
        # separate interacting proteins from non-interacting ones
        int_hpi_data = hpi_data[hpi_data['label'] == 1]
        nonint_hpi_data = hpi_data[hpi_data['label'] == 0]

        # Sample 10%, 25%, 50%, 75% of the original data
        percent = 100
        percent_frac = percent / 100
        int_hpi_data = int_hpi_data.sample(frac=percent_frac, random_state=seed)
        nonint_hpi_data = nonint_hpi_data.sample(frac=percent_frac, random_state=seed)
        hpi_data = pd.concat([int_hpi_data, nonint_hpi_data], axis=0, ignore_index=True)

        # separate features from label, group_id dataset
        X_hpi = hpi_data.drop(columns=['label', 'group_id'])
        y_hpi = hpi_data.label
        group_ids = hpi_data.group_id

        print("\nImbalanced dataset label count")
        print(y_hpi.value_counts())
        print(f"Sum of ids: {group_ids.value_counts().sum()}")

        # print(f"X: {X_hpi['group_id'].value_counts()}")
        # X_train, X_test, y_train, y_test = train_test_split(X_hpi, y_hpi, test_size=0.3, random_state=42)

    # The classifiers that would be trained
    clfs = [
        {'label': 'RF', 'model': RandomForestClassifier()},
        {'label': 'SVM', 'model': SVC()},
        {'label': 'MLP', 'model': MLPClassifier()},
        {'label': 'NB', 'model': GaussianNB()},
        {'label': 'LR', 'model': LogisticRegression()},
        {'label':'DF', 'model': CascadeForestClassifier()},
    ]

    # Output file header in excel
    worksheet.write(j, 0, fileTitle)
    worksheet.write(j + 1, 1, "Accuracy")
    worksheet.write(j + 1, 2, "Sensitivity")
    worksheet.write(j + 1, 3, "Specificity")
    worksheet.write(j + 1, 4, "Precision")
    worksheet.write(j + 1, 5, "F1 Score")
    worksheet.write(j + 1, 6, "MCC")
    worksheet.write(j + 1, 7, "AUROC")
    worksheet.write(j + 1, 8, "Time (sec)")
    worksheet.write(j + 1, 9, "Space (mb)")
    worksheet.write(j + 1, 10, "Accuracy_CI(+/-)")

    np.random.seed(seed)
    all_model_performance = []
    for clf in clfs:
        model_performance = []
        print(clf['label'] + " Model: RUNNING....")
        # Capture time and memory stating point
        # start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        tracemalloc.start()
        start_time = time.time()

        # replace NAN. INFINITY to Zero
        X_hpi = np.nan_to_num(X_hpi)

        # Define the model and scoring metrics
        model = make_pipeline(MinMaxScaler(), clf['model'])
        scoring = {
            'accuracy': make_scorer(custom_scorer, custom_metric='accuracy'),
            'sensitivity': make_scorer(custom_scorer, custom_metric='sensitivity'),
            'specificity': make_scorer(custom_scorer, custom_metric='specificity'),
            'precision': make_scorer(custom_scorer, custom_metric='precision'),
            'f1': make_scorer(custom_scorer, custom_metric='f1'),
            'mcc': make_scorer(custom_scorer, custom_metric='mcc'),
            'auroc': make_scorer(custom_scorer, custom_metric='auroc')
        }

        # Perform 10-fold cross-validation with scoring metrics
        # cv_results = cross_validate(model, X_hpi, y_hpi, cv=10, scoring=scoring)

        # Create Stratified Group K-Fold cross-validator
        cv = StratifiedGroupKFold(n_splits=10)

        # Perform cross-validation
        cv_results = cross_validate(model, X_hpi, y_hpi, groups=group_ids, cv=cv, scoring=scoring)


        # Capture time and memory end point and measure the difference
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

        # Save the trained model to disk
        model_filename = f"{clf['label']}_model_{file_tag}_{enc_type}.joblib"
        joblib.dump(model, outputPath + "diff_data_sizes/models/" + model_filename)
        print(f"Trained model saved as {model_filename}")


        # Measure predictive performance. Metrics to be measured are:
        # Accuracy, Sensitivity, Specificity, Precision, F1 Score, MCC and AUROC.
        # In addition, time and memory usage would also be measured

        # calculate accuracy standard error
        accuracy_std = cv_results['test_accuracy'].std()
        accuracy_stderr = accuracy_std / np.sqrt(len(cv_results['test_accuracy']))

        # Calculate accuracy confidence interval using the t-distribution
        confidence_level = 0.95
        dof = len(cv_results['test_accuracy']) - 1
        t_critical = t.ppf((1 + confidence_level) / 2, dof)
        accuracy_ci = t_critical * accuracy_stderr
        # accuracy_ci = (accuracy_mean - t_critical * accuracy_stderr, accuracy_mean + t_critical * accuracy_stderr)

        accuracy = cv_results['test_accuracy'].mean()
        sensitivity = cv_results['test_sensitivity'].mean()
        specificity = cv_results['test_specificity'].mean()
        precision = cv_results['test_precision'].mean()
        f1_score = cv_results['test_f1'].mean()
        mcc = cv_results['test_mcc'].mean()
        auroc = cv_results['test_auroc'].mean()

        worksheet.write(j + 2, 0, clf['label'])
        worksheet.write(j + 2, 1, accuracy)
        worksheet.write(j + 2, 2, sensitivity)
        worksheet.write(j + 2, 3, specificity)
        worksheet.write(j + 2, 4, precision)
        worksheet.write(j + 2, 5, f1_score)
        worksheet.write(j + 2, 6, mcc)
        worksheet.write(j + 2, 7, auroc)
        worksheet.write(j + 2, 8, end_time - start_time)
        worksheet.write(j + 2, 9, current / 10 ** 6)
        worksheet.write(j + 2, 10, accuracy_ci)

        model_performance = [accuracy, sensitivity, specificity, precision, f1_score, mcc, auroc]
        all_model_performance.append(model_performance)

        j = j + 1
        print(clf['label'] + " Model: FINISHED RUNNING!\n")
    j = j + 3
workbook.close()

# delete tmp directory
# deleteDir(tmp_dir)
print("Model Training Completed!")
print("Please check the file directory for results. Thank you!")

# Plot all model performance heatmap
sns.set(font='Arial', font_scale=1.4)  # Adjust font scale as needed
plt.figure(figsize=(11, 8.5), dpi=300)  # Set width, height for landscape orientation
ax = sns.heatmap(all_model_performance, annot=True, fmt='.3f', cmap='rocket_r', vmin=0, vmax=1)

ax.set_title('ML model performance - '+project_type_with_underscore+'\n');
ax.set_xlabel('\nPerformance metrics')
ax.set_ylabel('Machine Learning Models ');

ax.xaxis.set_ticklabels(['Accuracy','Sensitivity','Specificity','Precision','F1 Score','MCC','AUROC'])
ylabels = []
for clf in clfs: ylabels.append(clf['label'])
ax.yaxis.set_ticklabels(ylabels, rotation = 0)

# Save and Display the visualization of the Confusion Matrix.
plt.savefig(outputPath + "/diff_data_sizes/results/" + file_tag + "_" + enc_type + "_" + str(percent) +".pdf", format='pdf',
            bbox_inches='tight', orientation='landscape')
plt.show()