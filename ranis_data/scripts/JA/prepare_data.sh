SCRIPTS=scripts/JA
ORIGINALDATA=ranis/JA/data/goldstandard-v2/$1
OUTPUT=conv/JA/$1
# sentence split
mkdir -p ${OUTPUT}
cp ${ORIGINALDATA}/*.txt ${OUTPUT}
cp ${ORIGINALDATA}/*.ann ${OUTPUT}
echo "sentence split"
for i in ${OUTPUT}/*.txt
do
python ${SCRIPTS}/ssplit.py $i
done
# normalize
echo "normalize"
for i in ${OUTPUT}/*.split.txt
do
python ${SCRIPTS}/normalize.py $i > $i.normalized
mv $i.normalized $i
done
# move offset 
for i in ${OUTPUT}/*.split.txt
do
id=${OUTPUT}/`basename $i .split.txt`
python ${SCRIPTS}/standoff.py ${id}.txt ${id}.ann ${id}.split.txt > ${id}.split.ann
done
# gold so
echo "make gold.so"
for i in ${OUTPUT}/IPSJ-JNL*.split.txt
do
id=`basename $i .split.txt`
python ${SCRIPTS}/mkgold.py ${OUTPUT}/${id}.split.txt - ${OUTPUT}/${id}.split.ann ${id} > ${OUTPUT}/${id}.split.gold.so
done
# cabocha (=> *.cabocha.so)
echo "make cabocha.so"
for i in ${OUTPUT}/*.split.txt
do
sed -e "s/ /　/g" $i | cabocha -f1 -n1 > ${OUTPUT}/`basename $i .split.txt`.cabocha
done
for i in ${OUTPUT}/*.cabocha
do
id=${OUTPUT}/`basename $i .cabocha`
python ${SCRIPTS}/cabocha.py ${id}.cabocha ${id}.cabocha.txt ${id}.split.cabocha.so
done
