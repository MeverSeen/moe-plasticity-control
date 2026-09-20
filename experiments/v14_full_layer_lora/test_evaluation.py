import hashlib,json,unittest
from evaluation import GeneralEvaluator,compare_general,normalize_answer,token_f1
class Tests(unittest.TestCase):
 def test_manifest_permission_and_content_integrity(self):
  rows=[];manifest={'status':'passed','source_family_disjoint':True,'usage':'independent_evaluation_only','encoded_examples_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()};GeneralEvaluator(rows,manifest)
  with self.assertRaises(AssertionError):GeneralEvaluator([{'extra':'changed'}],manifest)
  with self.assertRaises(AssertionError):GeneralEvaluator(rows,dict(manifest,usage='training'))
 def test_general_delta_keeps_absolute_baselines_and_ids(self):
  initial={'manifest_sha256':'x','abilities':{'a':{'n':2,'ids':['x','y'],'correct':[True,False],'accuracy':.5,'answer_nll':2.}}};final={'manifest_sha256':'x','abilities':{'a':{'n':2,'ids':['x','y'],'correct':[False,False],'accuracy':0.,'answer_nll':3.}}};r=compare_general(initial,final)['a'];self.assertEqual(r['accuracy_delta'],-.5);self.assertEqual(r['initial_accuracy'],.5);self.assertEqual(r['answer_nll_delta'],1.)
  final['abilities']['a']['ids'].reverse()
  with self.assertRaises(AssertionError):compare_general(initial,final)
 def test_generation_metric_normalization(self):
  self.assertEqual(normalize_answer('The red, ball!'),'red ball');self.assertEqual(token_f1('red ball','the red ball'),1.);self.assertEqual(token_f1('','yes'),0.)
if __name__=='__main__':unittest.main()
