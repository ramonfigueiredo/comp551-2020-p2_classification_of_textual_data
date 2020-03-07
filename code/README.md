

## How to run the Python program?
1. Install [virtualenv](https://virtualenv.pypa.io/en/latest/)
	* To activate the virtualenv on Linux or MacOS: ```source venv/bin/activate```
	* To activate the virtualenv on Windows: ```\venv\Script\activate.bat```

2. Run the program

```sh
cd <folder_name>/

virtualenv venv -p python3 or python3 -m venv env  if you are using Mac



source venv/bin/activate

pip install -r requirements.txt

python main.py
```

**Note**: To desactivate the virtual environment

```sh
deactivate
```

For more help you can type ```python main.py -h```.

```
usage: main.py TODO
```


