import unittest
from main import user_input_features_test


class MyTestCase(unittest.TestCase):

    def test_user_input_feature_test(self):
        Age, GrLivArea, LotFrontage, LotArea, GarageArea, Fence, Pool = user_input_features_test()
        self.assert_(Age, 35)
        self.assert_(GrLivArea, 1522)
        self.assert_(LotFrontage, 70)
        self.assert_(LotArea, 10610)
        self.assert_(GarageArea, 478)
        self.assert_(Fence, True)
        self.assert_(Pool, False)


if __name__ == '__main__':
    unittest.main()
