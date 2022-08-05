<div align="center">

# Hyperbox


<a href="https://github.com/marsggbo/hyperbox"><img alt="Hyperbox" src="https://img.shields.io/badge/-Hyperbox-organe?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
<!-- <a href="https://hub.docker.com/r/ashlev/lightning-hydra"><img alt="Docker" src="https://img.shields.io/badge/docker-257bd6?style=for-the-badge&logo=docker&logoColor=white"></a> -->

A clean and scalable template to kickstart your AutoML project 🚀⚡🔥<br>
<!-- Click on [<kbd>Use this template</kbd>](https://github.com/ashleve/lightning-hydra-template/generate) to initialize new repository. -->

*Currently uses dev version of Hydra.<br>Suggestions are always welcome!*

</div>

You can refer to [Wiki](https://github.com/marsggbo/hyperbox/wiki) for more details.


<details>
<summary><b> Install </b></summary>

- install via `pip`
```
pip install hyperbox
```
- install via `github`

```
git clone https://github.com/marsggbo/hyperbox
cd hyperbox
python setup.py develop
python install -r requirements.txt
```

</details>

<details>
<summary><b> Quick Start </b></summary>

```
python -m hyperbox.run experiment=example_random_nas +trainer.fast_dev_run=True
```

</details>


<details>
<summary><b> Hyperbox Structure </b></summary>

- [Wiki for hyperbox.config](https://github.com/marsggbo/hyperbox/wiki/Customize-Config)
- [Wiki for hyperbox.mutables](https://github.com/marsggbo/hyperbox/wiki/Customize-Mutable)
- [Wiki for hyperbox.engine](https://github.com/marsggbo/hyperbox/wiki/Customize-Engine)
- [Wiki for hyperbox.mutator](https://github.com/marsggbo/hyperbox/wiki/Customize-Mutator)
- [Wiki for hyperbox.models](https://github.com/marsggbo/hyperbox/wiki/Customize-Models)
- [Wiki for hyperbox.networks](https://github.com/marsggbo/hyperbox/wiki/Customize-NAS-Network)
- [Wiki for Hydra](https://github.com/marsggbo/hyperbox/wiki/Hydra-Q&A)
- [Wiki for Hyperbox App](https://github.com/marsggbo/hyperbox/wiki/Hyperbox-App:-Start-a-new-project)
- [Miscellaneous](https://github.com/marsggbo/hyperbox/wiki/Miscellaneous-(tricks))
- [Q&A](https://github.com/marsggbo/hyperbox/wiki/Q&A)
- [Usage](https://github.com/marsggbo/hyperbox/wiki/Usages)

</details>


## Thanks

[![](https://shields.io/badge/-NNI-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/microsoft/nni/tree/v1.7)


[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/ashleve/lightning-hydra-template)   
