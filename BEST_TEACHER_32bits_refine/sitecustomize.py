import os
if os.environ.get('STRUCTURE32_EXPERIMENT')=='1':
    import setproctitle
    setproctitle.setproctitle('VLLM:Enginecore')
    setproctitle.setthreadtitle('VLLM:Enginecore')
