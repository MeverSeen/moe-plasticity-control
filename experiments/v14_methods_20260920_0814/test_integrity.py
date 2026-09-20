import unittest
from integrity import verify_frozen_hash
class Tests(unittest.TestCase):
 def test_structured_hash_matches_string_sha_and_parameter_scope(self):
  actual={'sha256':'abc','parameters':6919161856};self.assertNotEqual(actual,'abc');self.assertIs(verify_frozen_hash(actual,'abc'),actual)
  for invalid in [{'sha256':'different','parameters':6919161856},{'sha256':'abc','parameters':1},'abc']:
   with self.assertRaises(AssertionError):verify_frozen_hash(invalid,'abc')
if __name__=='__main__':unittest.main()
