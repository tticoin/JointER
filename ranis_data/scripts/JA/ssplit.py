# -*- coding: utf-8 -*-
import sys, re
import codecs

if len(sys.argv) != 2:
    sys.stderr.write('usage:' + sys.argv[0] + ' text')
    sys.exit(0)

out = codecs.open(re.sub('\.txt', '.split.txt', sys.argv[1]), 'w', 'utf-8')

for line in open(sys.argv[1]):
    line=re.sub(u'．', u'．\n', unicode(line, 'utf-8'))
    out.write(line)
