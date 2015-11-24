# JointER
Implementation of [Modeling Joint Entity and Relation Extraction with Table Representation](http://www.aclweb.org/anthology/D/D14/D14-1200.pdf) in EMNLP2014

## Prerequistites

* Sun JDK 1.7 or more
* ~20GB RAM (depends on corpus size)
* This software depends on the following libraries:
    * guava
    * jbzip2
    * mapdb
    * mtj
    * patricia-trie
    * snamekyaml

## Compilation

`ant jar`

## Sample usage on RANIS Japanese Corpus

### Data preparation 

* Prerequisites
    * mecab (with [neologd](https://github.com/neologd/mecab-ipadic-neologd))
    * cabocha 

* Data preparation

```sh
git submodule init
git submodule update
pushd ranis_data
bash scripts/JA/prepare_data.sh dev
bash scripts/JA/prepare_data.sh test
popd
```

### How to run 

* Relation Extraction

1. Train
`java -cp jointER.jar data.nlp.relation.RelationTrain yaml/parameters-ranis-ja-rel.yaml`
2. Test
`java -cp jointER.jar data.nlp.relation.RelationPredict yaml/parameters-ranis-ja-rel.yaml`

* Joint Entity and Relation Extraction

1. Train
`java -cp jointER.jar data.nlp.joint.JointTrain yaml/parameters-ranis-ja-joint.yaml`
2. Test
`java -cp jointER.jar data.nlp.joint.JointPredict yaml/parameters-ranis-ja-joint.yaml`

Please uncompress and use model files in ranis_data/pretrained if you want to use pretrained models.

## Sample usage on RANIS English Corpus

### Data preparation 

* Prerequisites
    * enju
    
* Data preparation

```sh
git submodule init # required if ranis is not updated
git submodule update # required if ranis is not updated
pushd ranis_data
bash scripts/EN/prepare_data.sh dev
bash scripts/EN/prepare_data.sh test
popd
```

Note: This package does not handle nested/disjoint entities and intersentential entities and relations.

### How to run 

* Joint Entity and Relation Extraction

1. Train
`java -cp jointER.jar data.nlp.joint.JointTrain yaml/parameters-ranis-en-joint.yaml`
2. Test
`java -cp jointER.jar data.nlp.joint.JointPredict yaml/parameters-ranis-en-joint.yaml`

## Configuration for new data

Please modify the yaml file.

## Notes

Please cite our paper when using this tool.
* Makoto Miwa and Yutaka Sasaki. Modeling Joint Entity and Relation Extraction with Table Representation. In the Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014). pp. 1858--1869, October 2014.

This work was supported by the TTI Start-Up Research Support Program and the JSPS Grant-in-Aid for Young Scientists (B) [grant number 25730129].

