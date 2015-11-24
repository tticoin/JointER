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

## Sample usage on RANIS JA

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

* Joint Relation Extraction

1. Train
`java -cp jointER.jar data.nlp.relation.RelationTrain yaml/parameters-ranis-ja-rel.yaml`
2. Test
`java -cp jointER.jar data.nlp.relation.RelationPredict yaml/parameters-ranis-ja-rel.yaml`

* Joint Entity and Relation Extraction

1. Train
`java -cp jointER.jar data.nlp.joint.JointTrain yaml/parameters-ranis-ja-joint.yaml`
2. Test
`java -cp jointER.jar data.nlp.joint.JointPredict yaml/parameters-ranis-ja-joint.yaml`

This package also contains the pipeline model in data.nlp.joint.pipeline package.

## Configuration for new data

Please modify the yaml file. For english, please refer to the useByte value and parser settings in the yaml/parameters-roth-joint.yaml

