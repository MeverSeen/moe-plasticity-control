import collections,unittest
from prepare_general import shingles,normalized,overlaps
class Tests(unittest.TestCase):
 def test_exhaustive_shingle_candidates_and_exact(self):
  rows=[' '.join('word%d'%i for i in range(100)),'unrelated tiny sentence here today'];index=collections.defaultdict(list);exact=collections.defaultdict(list);lengths=[]
  for i,row in enumerate(rows):
   ss=shingles(row);lengths.append(len(ss));exact[normalized(row)].append(i)
   for s in ss:index[s].append(i)
  self.assertEqual(overlaps(rows[0].upper(),index,lengths,exact)[0]['jaccard'],1.)
  near=rows[0]+' extra';matches=overlaps(near,index,lengths,exact);self.assertEqual(matches[0]['record'],0);self.assertGreater(matches[0]['jaccard'],.9)
  self.assertEqual(overlaps('a totally separate example of many different words',index,lengths,exact),[])
 def test_short_fragments_are_not_evidence_of_overlap(self):self.assertEqual(overlaps('yes',{},[],{'yes':[0]}),[])
if __name__=='__main__':unittest.main()
