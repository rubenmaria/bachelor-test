INPUT_DIR=$1
OUTPUT_NAME=$2
MODEL_PATH=models/c2vec-lite/saved_model_iter6.release
MODEL_DIRECTORY=models/c2vec-lite/
MAX_CONTEXTS=200
WORD_VOCAB_SIZE=1301136
PATH_VOCAB_SIZE=911417
TARGET_VOCAB_SIZE=261245
MAX_PATH_LENGTH=8
MAX_PATH_WIDTH=2
PYTHON=python
###########################################################
if [ "$#" -ne 2 ]; then
  echo "usage: generate-vectors-c [SRC_CODE] [OUTPUT_NAME]"
  exit 1
fi

TEST_DATA_FILE=${OUTPUT_NAME}.test.raw.txt
NAME_DATA_FILE=${OUTPUT_NAME}.names.txt
TARGET_HISTOGRAM_FILE=${OUTPUT_NAME}.histo.tgt.c2v
ORIGIN_HISTOGRAM_FILE=${OUTPUT_NAME}.histo.ori.c2v
PATH_HISTOGRAM_FILE=${OUTPUT_NAME}.histo.path.c2v

echo "Extracting paths from test set..."
${PYTHON} CExtractor/extract.py --dir ${INPUT_DIR} \
  --max_path_length ${MAX_PATH_LENGTH} --max_path_width ${MAX_PATH_WIDTH} \
  --ofile_name ${TEST_DATA_FILE} --ofile_name_names ${NAME_DATA_FILE}
echo "Finished extracting paths from test set"

echo "Creating histograms from the training data"
cat ${TEST_DATA_FILE} | cut -d' ' -f1 \
  | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${TEST_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 \
  | tr ',' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' \
  > ${ORIGIN_HISTOGRAM_FILE}
cat ${TEST_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 \
  | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${PATH_HISTOGRAM_FILE}

${PYTHON} preprocess.py --train_data ${TEST_DATA_FILE} \
  --test_data ${TEST_DATA_FILE} --val_data ${TEST_DATA_FILE} \
  --max_contexts ${MAX_CONTEXTS} --word_vocab_size ${WORD_VOCAB_SIZE} \
  --path_vocab_size ${PATH_VOCAB_SIZE} --target_vocab_size ${TARGET_VOCAB_SIZE}\
  --path_histogram ${PATH_HISTOGRAM_FILE}\
  --word_histogram ${ORIGIN_HISTOGRAM_FILE} \
  --target_histogram ${TARGET_HISTOGRAM_FILE} --output_name ${OUTPUT_NAME}
    
${PYTHON} code2vec.py --load ${MODEL_PATH} --export_code_vectors \
  --test ${OUTPUT_NAME}.test.c2v

# If all went well, the raw data files can be deleted, because preprocess.py
# creates new files with truncated and padded number of paths for each example.
rm ${TRAIN_DATA_FILE} ${VAL_DATA_FILE} ${TEST_DATA_FILE} \
  ${TARGET_HISTOGRAM_FILE} ${ORIGIN_HISTOGRAM_FILE} ${PATH_HISTOGRAM_FILE} \
  ${OUTPUT_NAME}.train.c2v ${OUTPUT_NAME}.val.c2v ${OUTPUT_NAME}.dict.c2v\
  ${OUTPUT_NAME}.test.c2v.num_examples ${OUTPUT_NAME}.test.c2v

