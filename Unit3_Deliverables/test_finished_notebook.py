"""Prevent errored or incomplete executions from becoming GitHub deliverables."""
import unittest
import nbformat
from download_finished_notebook import validate_finished_notebook


class FinishedNotebookTests(unittest.TestCase):
    def content(self, executed=True, error=False):
        """Construct nbformat's actual structured output variants, without logs."""
        cell = nbformat.v4.new_code_cell('print(1)', execution_count=1 if executed else None)
        if error:
            cell.outputs = [nbformat.v4.new_output('error', ename='ResourceExhaustedError', evalue='OOM', traceback=[])]
        else:
            cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text='1\n')]
        return nbformat.writes(nbformat.v4.new_notebook(cells=[cell])).encode('utf-8')

    def test_accepts_complete_notebook(self):
        self.assertEqual(validate_finished_notebook(self.content()).cells[0].execution_count, 1)

    def test_rejects_error_output(self):
        with self.assertRaises(ValueError):
            validate_finished_notebook(self.content(error=True))

    def test_rejects_unexecuted_cell(self):
        with self.assertRaises(ValueError):
            validate_finished_notebook(self.content(executed=False))


if __name__ == '__main__':
    unittest.main()
