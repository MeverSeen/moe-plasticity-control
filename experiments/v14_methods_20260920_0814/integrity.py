"""Compare the structured frozen-weight fingerprint without losing its metadata."""
def verify_frozen_hash(actual,expected_sha256,expected_parameters=6919161856):
 if not isinstance(actual,dict) or actual.get('sha256')!=expected_sha256 or actual.get('parameters')!=expected_parameters:
  raise AssertionError({'actual':actual,'expected_sha256':expected_sha256,'expected_parameters':expected_parameters})
 return actual
