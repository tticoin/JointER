# -*- coding: utf-8 -*-
import sys
import unicodedata
import codecs

if len(sys.argv) != 4:
    sys.stderr.write("usage:"+sys.argv[0]+" txt annotation newtxt > aligned_annotation")
    sys.exit(-1)

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
sys.stderr = codecs.getwriter('utf_8')(sys.stderr)

orig=[]
new=[]

for line in codecs.open(sys.argv[1], 'r', 'utf-8'):
    orig.append(line)

for line in codecs.open(sys.argv[3], 'r', 'utf-8'):
    new.append(line)

original="".join(orig)
newtext ="".join(new)

annotation=[]
terms = {}

for line in codecs.open(sys.argv[2], 'r', 'utf-8'):
    if line.startswith('T'):
        annots = line.rstrip().split("\t", 2)
        typeregion = annots[1].split(" ")
        start = int(typeregion[1])
        if not start in terms:
            terms[start] = []
        terms[start].append([start, int(typeregion[2])-start, annots[0], typeregion[0], ""])
    else:
        annotation.append(line)

orgidx = 0
newidx = 0
orglen = len(original)
newlen = len(newtext)

while orgidx < orglen and newidx < newlen:
    orgChar = unicodedata.normalize('NFKC', original[orgidx])
    newChar = unicodedata.normalize('NFKC', newtext[newidx])
    if orgChar == newChar:
        orgidx+=1
        newidx+=1
    elif newtext[newidx] == '\n':
        newidx+=1
    elif original[orgidx] == '\n':
        orgidx+=1
    elif newChar == u' ':
        newidx+=1
    elif orgChar == u' ':
        orgidx+=1
    else:
        try:
            assert False, "%d\t$%s$\t$%s$" % (orgidx, original[orgidx:orgidx+20], newtext[newidx:newidx+20])
        except:
            print "%d\t$%s$\t$%s$" % (orgidx, original[orgidx:orgidx+20], newtext[newidx:newidx+20])
            sys.exit(-1)
    if orgidx in terms: 
        for l in terms[orgidx]:
            l[0] = newidx

for ts in terms.values():
    for term in ts:
        #remove last space
        entity = newtext[term[0]:term[0]+term[1]].rstrip()
        print "%s\t%s %d %d\t%s" % (term[2], term[3], term[0], term[0]+len(entity), entity)

for annot in annotation:
    print annot,

