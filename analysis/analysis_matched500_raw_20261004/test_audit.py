import unittest
import audit_raw as a


class AuditTests(unittest.TestCase):
    def test_metrics_do_not_treat_lexical_acc_as_em(self):
        s = a.scores('It is Jessica Lange.', ['Jessica Lange'])
        self.assertEqual(s['answer_em'], 0)
        self.assertEqual(s['answer_acc'], 1)
        self.assertLess(s['answer_f1'], 1)

    def test_oracle_uses_same_metric_then_cost(self):
        rs = {('q', '100'): {'scores': {'answer_em': 0, 'answer_f1': .8}, 'total_tokens': 10},
              ('q', '200'): {'scores': {'answer_em': 1, 'answer_f1': 1}, 'total_tokens': 100},
              ('q', 'FULL'): {'scores': {'answer_em': 1, 'answer_f1': 1}, 'total_tokens': 200}}
        self.assertEqual(a.select(rs, 'q', ['100', '200', 'FULL'], 'answer_em'), '200')

    def test_pair_matrix_direction(self):
        m = a.paired.binary_matrix([True, True, False, False], [True, False, True, False])
        self.assertEqual(m['correct_to_wrong'], 1)
        self.assertEqual(m['wrong_to_correct'], 1)
        self.assertEqual(m['after_correct'], 2)

    def test_unicode_text_difference_can_leave_tokens_identical(self):
        x, y = 'Pokémon', 'Poke\u0301mon'
        self.assertNotEqual(a.sha(x), a.sha(y))
        self.assertEqual(a.ids_hash(x), a.ids_hash(y))


if __name__ == '__main__':
    unittest.main()
