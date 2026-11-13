"""Apply the project's requested process title only to this isolated launch."""
import setproctitle
setproctitle.setproctitle('VLLM::Enginecore')
setproctitle.setthreadtitle('VLLM::Enginecore')
