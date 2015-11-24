# -*- coding: utf-8 -*-
import sys, re
import codecs

if len(sys.argv) < 3:
    sys.stderr.write("python %s cabocha txt dep" % (sys.argv[0]))
    sys.exit(-1)


phrases = []
headfuncs = {}
dep = {}
offset = 0

phrase_id = 0
word_id = 0
sent_id = 0

id_idx = 0
text_out = codecs.open(sys.argv[2], 'w', 'utf-8')
dep_out = codecs.open(sys.argv[3], 'w', 'utf-8')
nlines = 0
PAS_ID={}

for line in codecs.open(sys.argv[1], 'r', 'utf-8'):
    nlines += 1
    if line.startswith('* '):
        ph_tag = line.split(' ')
        assert len(ph_tag) > 2, str(nlines)+line
        if ph_tag[2].endswith('D'):
            dep[int(ph_tag[1])] = int(ph_tag[2][0:-1])
        head_func = ph_tag[3].split("/")
        headfuncs[phrase_id+int(ph_tag[1])] = [int(head_func[0]), int(head_func[1])]
        phrases.append([])
    elif line.startswith('EOS'):
        sent_len = 0
        content = []
        tmp_wid = word_id
        for phrase in phrases:
            for word in phrase:
                content.append(word[0])
                sent_len += len(word[0])
                if u"id" in word[5]:
                    PAS_ID[int(word[5][u"id"])] = tmp_wid+1
                tmp_wid+=1
        content.append('\n')
        text_out.write("".join(content))
        dep_out.write('%s\t%s\tsentence\tid="%s"\n' % (offset, offset+sent_len, "s"+str(sent_id+1)))
        pid = 0
        pbase = phrase_id        
        heads = {}
        tmp_phrase_id = phrase_id
        tmp_wid = word_id
        for phrase in phrases:
            nwords = len(phrase)
            if tmp_phrase_id in headfuncs:
                head_idx = headfuncs[tmp_phrase_id][0]
                #if head_idx > 0 and phrase[head_idx][3] == u"接尾":
                #    head_idx = head_idx - 1
                #if phrase[head_idx][3] == u"自立" and phrase[head_idx][1] == u"動詞" and phrase[head_idx][2] == u"する" and head_idx > 0 and phrase[head_idx-1][3] == u"サ変接続":
                #    head_idx = head_idx - 1
                heads[tmp_phrase_id] = tmp_wid+head_idx+1
            else:
                heads[tmp_phrase_id] = tmp_wid+1
            tmp_wid += nwords
            tmp_phrase_id += 1
        for phrase in phrases:
            phrase_len = 0
            to_dep = None
            if pid in dep and dep[pid] >= 0:
                to_dep = dep[pid] + pbase
            nwords = len(phrase)
            for word in phrase:
                phrase_len += len(word[0])
            hw = heads[phrase_id]
            func = 0 
            if phrase_id in heads:
                func = heads[phrase_id]
            else:
                func = 1+word_id
            dep_out.write('%s\t%s\tcons\tid="%s" head="%s"\n' % (offset, offset+phrase_len, "c"+str(phrase_id+1), "t"+str(heads[phrase_id])))
            for idx in range(nwords):
                word = phrase[idx]
                word_len = len(word[0])
                if word_id + 1 == hw and hw != func:
                    dep_out.write('%s\t%s\ttok\tid="%s" base="%s" pos="%s" cat="%s" NMOD="%s"' % (offset, offset+word_len, "t"+str(word_id+1), word[2], word[1], word[3], "t"+str(func)))
                elif word_id + 1 == func:
                    if to_dep is None:
                        dep_out.write('%s\t%s\ttok\tid="%s" base="%s" pos="%s" cat="%s" ROOT="ROOT"' % (offset, offset+word_len, "t"+str(word_id+1), word[2], word[1], word[3]))
                    else:
                        #dep_out.write('%s\t%s\ttok\tid="%s" base="%s" pos="%s" cat="%s" NMOD="%s"' % (offset, offset+word_len, "t"+str(word_id+1), word[2], word[1], word[3], "c"+str(to_dep+1)))
                        assert to_dep in heads
                        dep_out.write('%s\t%s\ttok\tid="%s" base="%s" pos="%s" cat="%s" NMOD="%s"' % (offset, offset+word_len, "t"+str(word_id+1), word[2], word[1], word[3], "t"+str(heads[to_dep])))
                else:
                    if word_id + 1 < hw:
                        dep_out.write('%s\t%s\ttok\tid="%s" base="%s" pos="%s" cat="%s" NMOD="%s"' % (offset, offset+word_len, "t"+str(word_id+1), word[2], word[1], word[3], "t"+str(hw)))
                    else:
                        dep_out.write('%s\t%s\ttok\tid="%s" base="%s" pos="%s" cat="%s" NMOD="%s"' % (offset, offset+word_len, "t"+str(word_id+1), word[2], word[1], word[3], "t"+str(func)))
                if word[4] != "O":
                    dep_out.write(' NE="%s"' % (word[4]))
                if len(word[5]) != 0:
                    type=""
                    if u"type" in word[5]:
                        type = word[5][u"type"]
                    for k,v in word[5].items():
                        if k == u"type" or k == u"id":
                            continue
                        # TODO: inter-sentential "eq" 
                        if int(v.strip()) in PAS_ID:
                            if k == u"eq":
                                continue
                                #dep_out.write(' %s="%s"' % (k,"t"+str(PAS_ID[int(v)])))
                            else:
                                dep_out.write(' %s="%s"' % (type+"_"+k,"t"+str(PAS_ID[int(v)])))
                dep_out.write('\n')
                offset += word_len
                word_id += 1
            phrase_id += 1
            pid += 1
        phrases = []
        headfuncs = {}
        dep = {}
        sent_id += 1
        offset += 1
    elif line.startswith('EOT'):
        continue
    else:
        word_annot = line.split('\t')
        word_tag = re.sub(u',', u'\t', word_annot[1]).split('\t')
        if word_tag[6] == "*" and word_annot[0] != "*":
            word_tag[6] = word_annot[0]
        pas={}
        if word_annot[2][-1] == '\n':
            word_annot[2] = word_annot[2][0:-1]
        word = (word_annot[0], word_tag[0], word_tag[6], word_tag[1], word_annot[2], pas) # surface, pos, base, pos2, NE, PAS
        phrases[-1].append(word)


text_out.close()
dep_out.close()



