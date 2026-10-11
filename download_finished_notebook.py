"""Download only a successfully executed Kaggle notebook, never model weights.

Use --kernel owner/slug --version N. Both Kaggle's structured COMPLETE status and
nbformat's execution/error fields must pass before any file enters kaggle_results.
Failed runs must be debugged and relaunched instead of archived as finished.
"""
import argparse
from pathlib import Path
import nbformat


def validate_finished_notebook(content):
    """Validate nbformat structure and every code cell's execution result.

    Standard code cells expose execution_count and outputs. Error outputs expose
    output_type='error', ename, evalue, traceback; any error rejects the artifact.
    Other output types are preserved verbatim. Metadata and markdown are retained.
    """
    notebook = nbformat.reads(content.decode('utf-8'), as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    if not code_cells or any(cell.execution_count is None for cell in code_cells):
        raise ValueError('Notebook has unexecuted code cells; no finished archive will be written.')
    for cell in code_cells:
        for output in cell.outputs:
            if output.output_type == 'error':
                raise ValueError(f'Notebook execution failed: {output.ename}: {output.evalue}')
    return notebook


def download_finished_notebook(kernel, version, destination):
    """Use typed Kaggle requests with pinned version, then atomically save bytes.

    Metadata's current_version_number must match the requested version: Kaggle's
    status endpoint accepts the current slug but rejects numeric version labels.
    Status response fields: status is compared to COMPLETE; failure_message is
    surfaced if rejected. Download returns an HTTP Response in the installed SDK;
    raise_for_status checks transport failures and content supplies notebook bytes.
    Remaining HTTP headers are not needed. No output bundle or weights are fetched.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelRequest, ApiGetKernelSessionStatusRequest, ApiDownloadKernelOutputRequest)
    from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus

    owner, slug = kernel.split('/')
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        metadata_request = ApiGetKernelRequest()
        metadata_request.user_name = owner
        metadata_request.kernel_slug = slug
        metadata = client.kernels.kernels_api_client.get_kernel(metadata_request).metadata
        if metadata.current_version_number != version:
            raise RuntimeError('Only the current version can be status-verified; specify its exact version number.')
        request = ApiGetKernelSessionStatusRequest()
        request.user_name = owner
        request.kernel_slug = slug
        status = client.kernels.kernels_api_client.get_kernel_session_status(request)
        if status.status != KernelWorkerStatus.COMPLETE:
            raise RuntimeError(f'Refusing archive: {status.status.name}. {status.failure_message}')
        download = ApiDownloadKernelOutputRequest()
        download.owner_slug = owner
        download.kernel_slug = slug
        download.version_number = version
        download.file_path = '__notebook__.ipynb'
        response = client.kernels.kernels_api_client.download_kernel_output(download)
        response.raise_for_status()
        content = response.content
    notebook = validate_finished_notebook(content)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.ipynb.tmp')
    temporary.write_bytes(content)
    temporary.replace(destination)
    print(f'Saved successful notebook: {destination} ({len(notebook.cells)} cells)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--version', required=True, type=int)
    parser.add_argument('--destination', default='kaggle_results/Notebook_finished.ipynb')
    args = parser.parse_args()
    download_finished_notebook(args.kernel, args.version, args.destination)
