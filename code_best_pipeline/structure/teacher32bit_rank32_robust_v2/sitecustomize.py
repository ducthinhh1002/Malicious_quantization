"""Requested display name; campaign identity remains explicit in logs/manifest."""
import setproctitle
setproctitle.setproctitle('VLLM::Enginecore')
setproctitle.setthreadtitle('VLLM::Enginecore')

