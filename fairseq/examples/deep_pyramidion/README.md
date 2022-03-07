
## Setup
We provide a code implementation that is based on fairseq library. 
We forked from `fairseq 0.10.2` and directly modified some of its files and added a few more.
The library should be unpacked and installed from requirements.
It requires:
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)


## Prepare data
1. Download original data with `data/download_arxiv.sh`
2. Flatten jsons into text with `data/preprocess_arxiv.py`

### Encode raw into BPE
```shell
DATA_FOLDER=./data/summarization

for set in test-A train dev-0
do
        mkdir ./data/bpe/${set}/
        python fairseq/scripts/spm_encode.py \
        --model spm/spm.model \
        --inputs ./data/raw/${set}/in.tsv ./data/raw/${set}/expected.tsv \
        --outputs \
                ./data/bpe/${set}/in.tsv \
                ./data/bpe/${set}/expected.tsv
done
```
### Binarize bpe into bin

```shell
for set in train dev-0 test-A
do
        echo $set
        for file in in.tsv expected.tsv 
        do 
                file_short="expected"
                if [ $file = "in.tsv" ]; then
                        file_short="in"
                fi
                echo $file_short
                fairseq-preprocess\
                        --destdir ./data/bin/${set}/\
                        --joined-dictionary\
                        --task\
                        translation\
                        --dataset-impl\
                        mmap\
                        --workers\
                        16\
                        --seed\
                        31337\
                        --srcdict\
                        ./spm/spm.vocab\
                        --only-source\
                        --trainpref\
                        ./data/bpe/${set}/${file}
                mv ./data/bin/${set}/train.bin   ./data/bin/${set}/${file_short}.bin  
                mv ./data/bin/${set}/train.idx   ./data/bin/${set}/${file_short}.idx
        done
        mv ./data/bin/${set}/in.bin  ./data/summarization/${set}.src-tgt.src.bin  
        mv ./data/bin/${set}/in.idx   ./data/summarization/${set}.src-tgt.src.idx
        mv ./data/bin/${set}/expected.bin  ./data/summarization/${set}.src-tgt.tgt.bin  
        mv ./data/bin/${set}/expected.idx   ./data/summarization/${set}.src-tgt.tgt.idx
done


mv ./data/bin/${set}/dict.txt  ./data/summarization/dict.src.txt
mv ./data/bin/${set}/dict.txt  ./data/summarization/dict.tgt.txt      
```

### Prepare out for validation during training
```shell
cp ./data/bpe/dev-0/expected.tsv ./data/summarization/dev-0-gold.tsv
sed "s/ //g" -i  ./data/summarization/dev-0-gold.tsv  | sed "s/▁/ /g" -i ./data/summarization/dev-0-gold.tsv
```

```shell
EXP_NAME=8    # Must be in range {1..33}. It corresponds to the paper's ordering.
TRAIN_ARGS=
CUDA_VISIBLE_DEVICES=${DEVICE} fairseq-train 
        ${DATA_FOLDER} \
        -c ./configs/${EXP_NAME}_config.yaml \
        --arch transformer \
        --task translation \
        --valid-subset ${DATA_FOLDER}/dev-0 \
        --save-dir results/train/${EXP_NAME} \
        --dataset-impl mmap
```

## Pretrained models 
Pretrained models will be released shortly. 

