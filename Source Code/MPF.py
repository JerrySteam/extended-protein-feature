from module import *

# Project/dataset directory path
dataset_type = "bacteria/bacillus" # bacteria/bacillus, virus/human_sars_cov2, virus/human_hiv1, parasite/plasmodium
                        # plant/Ara_PsyGorHpa Oomycetes/Ara-Hpa, Bacteria/Ara-Psy, Combined_HPI/Ara_PsyGorHpa
main_path = "C:/Users/Jerry/Desktop/Implementation/"+dataset_type+"/"
# main_path = "C:/Users/Jerry/Desktop/Itunu/"+dataset_type+"/"
dataset_type_with_underscore = str(dataset_type.split("/")[0]) + "_" + str(dataset_type.split("/")[1])
file_tag = str(time.time()).split(".")[0]

# create temporary directory for storing files if it does not exist
tmp_dir = createDir(main_path+"tmp/")

# ===================Data Preprocessing starts here =============================
print("Preprocessing Starts...")
# read dataset files (fasta format) from the project directory into list
int_host = readFasta(main_path+"inthost.fasta")
int_pathogen = readFasta(main_path+"intpathogen.fasta")
nonint_host = readFasta(main_path+"noninthost.fasta")
nonint_pathogen = readFasta(main_path+"nonintpathogen.fasta")

# delete any invalid protein sequence
print("Removing invalid protein pairs...")
int_host, int_pathogen = deleteInvalidProt(int_host, int_pathogen, "interacting")
nonint_host, nonint_pathogen = deleteInvalidProt(nonint_host, nonint_pathogen, "noninteracting")

# ensure same length for nonintprot1 and nonintprot2
# nonint_pathogen = sample(nonint_pathogen, len(nonint_host))

# ==================================================
concat_int = []
concat_nonint = []
for i in range(len(int_host)):
    id = int_host[i][0]+"_"+int_pathogen[i][0]
    seq = int_host[i][1]+""+int_pathogen[i][1]
    concat_int.append([id, seq])

for i in range(len(nonint_host)):
    id = nonint_host[i][0]+"_"+nonint_pathogen[i][0]
    seq = nonint_host[i][1]+""+nonint_pathogen[i][1]
    concat_nonint.append([id, seq])

# set output file path for the feature vectors
int_path = tmp_dir+"int_"+file_tag
nonint_path = tmp_dir+"nonint_"+file_tag

# perform the feature encoding technique (Note: This operations takes time)
print("Performing protein sequence feature encoding...")
intx_path = generateFeatures(convertToFasta(concat_int, int_path+".fasta"), int_path+"_x.csv")
nonintx_path = generateFeatures(convertToFasta(concat_nonint, nonint_path+".fasta"), nonint_path+"_x.csv")

# Remove index column from files
intx = pd.read_csv(intx_path, delimiter='\t', encoding='latin-1').iloc[: , 1:]
nonintx = pd.read_csv(nonintx_path, delimiter='\t', encoding='latin-1').iloc[: , 1:]

# add label feature
int_hp = intx
nonint_hp = nonintx
# ==================================================

int_hp['label'] = 1
nonint_hp['label'] = 0

# combine interacting and non interacting protein
hpi_data = int_hp.append(nonint_hp, ignore_index=True)

# save extracted features to csv
preprocessedPath = createDir(main_path + "preprocessed/")
hpi_data.to_csv(preprocessedPath + dataset_type_with_underscore + ".csv")
print("Preprocessing Completed!")
# ===================end of data preprocessing =============================

# ===================Computational model starts =============================
print("Model Training Starts...")
# count number of interactions and non-interactions
print(hpi_data['label'].value_counts())
print(hpi_data.shape)

# Output path directory
outputPath = createDir(main_path + "output/")

# Create output file in excel
workbook = xlsxwriter.Workbook(outputPath + dataset_type_with_underscore + file_tag + ".xlsx")
worksheet = workbook.add_worksheet()

# Run experiment for both balanced and imbalanced dataset i.e range(0,2).
# If only balance, set range(0,1), if only imbalance, set range (1,2)
j = 0
for count in range(0, 1):
    if count == 0: analysis_type = "balanced"
    else: analysis_type = "imbalanced"
    fileTitle = str(analysis_type) + "_dataset_" + str(dataset_type_with_underscore)

    if analysis_type == "balanced":
        # separate interacting proteins from non-interacting ones
        int_hpi_data = hpi_data[hpi_data['label'] == 1]
        nonint_hpi_data = hpi_data[hpi_data['label'] == 0]

        # print("\nInteracting protein label count")
        print("\nBalanced dataset label count")
        if len(int_hpi_data) >= len(nonint_hpi_data):
            # Create balance dataset: Use this when interacting is more than non-interacting samples
            # randomly select interacting samples
            rand_int_hpi_data = int_hpi_data.sample(len(nonint_hpi_data))

            # merge the nonint random samples with the int samples to form a balance dataset
            bal_hpi_data = rand_int_hpi_data.append(nonint_hpi_data, ignore_index=True)
            # bal_hpi_data
            # print(bal_hpi_data['label'].value_counts())

            # separate balanced hpi features from labels
            X_hpi = bal_hpi_data.drop(columns='label')
            y_hpi = bal_hpi_data.label
            print(y_hpi.value_counts())

        else:
            # Create balance dataset: Use this when non-interacting is more than interacting samples
            # randomly select non-interacting samples
            rand_nonint_hpi_data = nonint_hpi_data.sample(len(int_hpi_data), replace=False)

            # merge the nonint random samples with the int samples to form a balance dataset
            bal_hpi_data = int_hpi_data.append(rand_nonint_hpi_data, ignore_index=True)
            # bal_hpi_data
            # print(bal_hpi_data['label'].value_counts())

            # separate balanced hpi features from labels
            X_hpi = bal_hpi_data.drop(columns='label')
            y_hpi = bal_hpi_data.label
            print(y_hpi.value_counts())

    else:
        # Create imbalance dataset
        # separate imbalanced hpi features from labels
        X_hpi = hpi_data.drop(columns='label')
        y_hpi = hpi_data.label
        # print("\nNon-interacting protein label count")
        print("\nImbalanced dataset label count")
        print(y_hpi.value_counts())

    # create train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X_hpi, y_hpi, test_size=0.30)
    # np.count_nonzero(y_test == 0)
    print("\nTraining dataset label count")
    print(y_train.value_counts())

    print("\nTest dataset label count")
    print(y_test.value_counts())

    # normalize the features of train and test
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # pd.DataFrame(X_train)

    # The classifiers that would be trained
    clfs = [
        {'label': 'RF', 'model': RandomForestClassifier()},
        {'label': 'SVM', 'model': SVC()},
        {'label': 'MLP', 'model': MLPClassifier()},
        {'label': 'NB', 'model': GaussianNB()},
        {'label': 'LR', 'model': LogisticRegression()},
        {'label':'DF', 'model': CascadeForestClassifier()},
        # {'label':'LSTM', 'model': KerasClassifier(model=lstm_baseline_model, epochs=100, batch_size=5, verbose=0)},
        # {'label': 'LSTM', 'model': KerasClassifier(model=lstm_baseline_model(X_train))},
    ]

    # Output file header in excel
    worksheet.write(j, 0, fileTitle)
    worksheet.write(j+1, 1, "Accuracy")
    worksheet.write(j+1, 2, "Sensitivity")
    worksheet.write(j+1, 3, "Specificity")
    worksheet.write(j+1, 4, "Precision")
    worksheet.write(j+1, 5, "F1 Score")
    worksheet.write(j+1, 6, "MCC")
    worksheet.write(j+1, 7, "AUROC")
    worksheet.write(j+1, 8, "Time (sec)")
    worksheet.write(j+1, 9, "Space (mb)")

    np.random.seed(10)
    all_model_performance = []
    for clf in clfs:
        model_performance = []
        print(clf['label'] + " Model: RUNNING....")
        # Capture time and memory stating point
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        start_time = time.time()

        # replace NAN. INFINITY to Zero
        X_train = np.nan_to_num(X_train)

        # Train and Test the models
        model = clf['model']
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Capture time and memory end point and measure the difference
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

        # Measure predictive performance of the models on test data and write output to excel. Metrics to be measured are:
        # Accuracy, Sensitivity, Specificity, Precision, F1 Score, Matthew's Correlation Coefficient (MCC) and AUROC.
        # In addition, time and memory usage would also be measured
        accuracy = metrics.accuracy_score(y_test, y_pred)
        sensitivity = metrics.recall_score(y_test, y_pred)
        specificity = metrics.recall_score(y_test, y_pred, pos_label=0)
        precision = metrics.precision_score(y_test, y_pred)
        f1_score = metrics.f1_score(y_test, y_pred)
        mcc = metrics.matthews_corrcoef(y_test, y_pred)
        auroc = metrics.roc_auc_score(y_test, y_pred)

        worksheet.write(j+2, 0, clf['label'])
        worksheet.write(j+2, 1, accuracy)
        worksheet.write(j+2, 2, sensitivity)
        worksheet.write(j+2, 3, specificity)
        worksheet.write(j+2, 4, precision)
        worksheet.write(j+2, 5, f1_score)
        worksheet.write(j+2, 6, mcc)
        worksheet.write(j+2, 7, auroc)
        worksheet.write(j+2, 8, end_time-start_time)
        worksheet.write(j+2, 9, end_memory - start_memory)

        model_performance = [accuracy, sensitivity, specificity, precision, f1_score, mcc, auroc]
        all_model_performance.append(model_performance)

        j = j + 1
        print(clf['label'] + " Model: FINISHED RUNNING!\n")
    j = j + 3
workbook.close()

# Plot all model performance heatmap
# plt.figure(figsize = (7,7))
ax = sns.heatmap(all_model_performance, annot=True, fmt='.4f', cmap='rocket_r', vmin=0, vmax=1)

ax.set_title('ML model performance - '+dataset_type_with_underscore+'\n');
ax.set_xlabel('\nPerformance metrics')
ax.set_ylabel('Machine Learning Models ');

ax.xaxis.set_ticklabels(['Accuracy','Sensitivity','Specificity','Precision','F1 Score','MCC','AUROC'])
ylabels = []
for clf in clfs: ylabels.append(clf['label'])
ax.yaxis.set_ticklabels(ylabels, rotation = 0)

# Display the visualization of the Confusion Matrix.
plt.show()


# delete tmp directory
deleteDir(tmp_dir)
print("Model Training Completed!")
print("Please check the file directory for results. Thank you!")