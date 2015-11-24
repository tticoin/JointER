# -*- coding: utf-8 -*-
import sys

tags = []
index = 0
ids = {}

ignore_term=[]#%"Not_sure_entity","Other_entity","AnnotatorNotes","TODO","TODO-SMP","frag","Inhibition","Sub-Process","Other_event","Core_Angiogenesis_Term","Protein_family_or_group"]
ignore_event=[]#"Inhibition","Sub-Process","Other_event","TODO","TODO-SMP"]

for line in open(sys.argv[1]):
    line = unicode(line, 'utf-8')
    tags.append([index, index+len(line)-1, "sentence"])
    index = index + len(line)
tags.append([0, index, "article\tpmid=\"%s\"" % sys.argv[4]])

pair=1
trigger = {}
a2file = []
if sys.argv[3] != "-":
    a2file = open(sys.argv[3])

hedges = {}
equivs = {}
for line in a2file:
    if line.startswith("*"):
        a2 = line.strip().split('\t')
        eq_list = a2[1].split(" ")
        if eq_list[0] == "Equiv":
            if not equivs.has_key(eq_list[1]):
                equivs[eq_list[1]] = []
            for i in range(2, len(eq_list)):
                equivs[eq_list[1]].append(eq_list[i])
    elif line.startswith("M") or line.startswith("A"): 
        a2 = line.strip().split('\t')
        hedge_list = a2[1].split(" ")
        if not hedges.has_key(hedge_list[1]):
            hedges[hedge_list[1]] = []
        hedges[hedge_list[1]].append(hedge_list[0])

if sys.argv[2] != "-":
    for line in open(sys.argv[2]):
        a1 = line.strip().split('\t')
        [type, start, end] = a1[1].split(" ")
        start = int(start)
        end = int(end)
        if type in ignore_term:
            continue
        ids[a1[0]] = [type, start, end]
        if len(a1) > 3:
            prot_tag = "%s\tid=\"%s\" %s" % (type, a1[0], a1[3])
        else:
            prot_tag = "%s\tid=\"%s\"" % (type, a1[0])
        if equivs.has_key(a1[0]):
            for equiv in equivs[a1[0]]:
                prot_tag += " Equiv=\"%s\"" % (equiv)
        tags.append([start, end, prot_tag])
    
if sys.argv[3] != "-":
    a2file.close()
    a2file = open(sys.argv[3])

for line in a2file:
    if line.startswith("T"):
        a2 = line.strip().split('\t')
        if len(a2) < 2:
            continue
        if len(a2[1].split(" ")) != 3:
            continue
        [type, start, end] = a2[1].split(" ")
        start = int(start)
        end = int(end)
        if type in ignore_term:
            continue
        if len(a2) > 2 and a2[2].endswith("."): 
            end -= 1 #for sentence annotation
        ids[a2[0]] = [type, start, end]
        tags.append([start, end, "%s\tid=\"%s\"" % (type, a2[0])])

    
    
if sys.argv[3] != "-":
    a2file.close()
    a2file = open(sys.argv[3])

event_tags=[]        
for line in a2file:
    if line.startswith("E"):
        a2 = line.strip().split('\t')
        a2type=a2[1].split(" ")
        triggerid=a2type[0].split(":")[1]
        try:
            event=ids[triggerid]
        except:
            sys.stderr.write(triggerid)
            sys.stderr.write('\n')
            continue
        if event[0] in ignore_event:
            continue
        assert event[0] == a2type[0].split(":")[0], "%s" % (triggerid)
        event_tag = "Event\tpair=\"%d\" type=\"%s\" id=\"%s\"" % (pair, event[0], a2[0])
        if hedges.has_key(a2[0]):
            for hedge_type in hedges[a2[0]]:
                event_tag += " hedge=\"%s\" " % (hedge_type)
        ev_tags = []
        ev_unknown = []
        ev_tags.append([event[1], event[2], event_tag])
        for arg in a2type[1:]:
            if len(arg) == 0:
                continue
            type = arg.split(":")[0]
            id = arg.split(":")[1]
            if trigger.has_key(id):
                id = trigger[id]
            if ids.has_key(id):
                argument=ids[id]
                ev_tags.append([argument[1], argument[2], "%s\tpair=\"%d\" type=\"%s\"" % (type, pair, argument[0])])
            else:
                ev_unknown.append([id, pair, type])
        if len(ev_unknown) == 0:
            trigger[a2[0]]=triggerid
            tags.extend(ev_tags)
        else:
            event_tags.append((a2[0],triggerid,ev_tags,ev_unknown))
        pair += 1
    elif line.startswith("*") or line.startswith("M") or line.startswith("T") or line.startswith("A"):
        continue
    elif line.startswith("R"):
        # skip relation annotation
        continue   

lastsize = -1
while lastsize != len(trigger):
    lastsize = len(trigger)
    for eventid,triggerid,ev_tags,ev_unknown in event_tags:
        if eventid in trigger:
            continue
        no_unknown = True
        for rest in ev_unknown:
            [id, pair, type] = rest
            if trigger.has_key(id):
                id = trigger[id]
            if ids.has_key(id):
                argument=ids[id]
                ev_tags.append([argument[1], argument[2], "%s\tpair=\"%d\" type=\"%s\"" % (type, pair, argument[0])])
            else:
                no_unknown = False
                break
        if no_unknown:
            trigger[eventid]=triggerid
            tags.extend(ev_tags)

def listsort(x, y):
    if x[0] < y[0]:
        return -4
    elif x[0] > y[0]:
        return +4
    elif x[1] > y[1]:
        return -2
    elif x[1] < y[1]:
        return +2
    elif cmp(x[2], y[2]) < 0:
        return -1
    elif cmp(x[2], y[2]) > 0:
        return +1
    else:
        return 0


tags.sort(lambda x, y: listsort(x, y))

for tag in tags:
    print "%d\t%d\t%s" % (tag[0], tag[1], tag[2])
    
