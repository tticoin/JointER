function pwait() {
    while [ `jobs -p | wc -l` -ge $1 ]; do
        sleep 1
    done
}
PROCS=4
SCRIPTS=scripts/EN
ORIGINALDATA=ranis/EN/data/original/$1
OUTPUT=conv/EN/$1
# copy files
mkdir -p ${OUTPUT}
cp ${ORIGINALDATA}/**/*.txt ${OUTPUT}
cp ${ORIGINALDATA}/**/*.ann ${OUTPUT}
# remove disjoint entity
sed -i -e 's/([0-9]+ )([0-9]+;[0-9]+ )+([0-9]+)/\1\3/g' -e "/^[^TR].*/d" ${OUTPUT}/*.ann
# parse with enju
for i in ${OUTPUT}/*.txt
do
pwait ${PROCS}    
enju -so -W 200 < $i > ${OUTPUT}/`basename $i .txt`.enju.so 2> /dev/null &
done
wait


