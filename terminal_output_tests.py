#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from clint.arguments import Args
from clint.textui import puts, colored, indent

args = Args()

with indent(4, quote='>>>'):
    puts(colored.blue('Aruments passed in: ') + str(args.all))
    puts(colored.blue('Flags detected: ') + str(args.flags))
    puts(colored.blue('Files detected: ') + str(args.files))
    puts(colored.blue('NOT Files detected: ') + str(args.not_files))
    puts(colored.blue('Grouped Arguments: ') + str(dict(args.grouped)))

print()

# use colorful
import colorful as cf

# use colorama
from colorama import Fore, Back, Style

# use termcolor
from termcolor import colored

# using colorama to print out red text
print(Fore.RED + 'some red text')
print(Back.GREEN + 'and with a green background')
print(Style.DIM + 'and in dim text')


from rich.progress import track
from time import sleep

def scrape_data():
    sleep(0.1)

for _ in track(range(100), description='[green]Scraping data'):
    scrape_data()


# or to get the time a task is finished use this code

from rich.console import Console
from time import sleep

console = Console()

datas = [1,2,3,4,5]
with console.status("[bold green]Scraping data...") as status:
    while datas:
        data = datas.pop(0)
        sleep(1)
        console.log(f"[green]Finish scraping data[/green] {data}")

    console.log(f'[bold][red]Done!')



from termcolor import colored
from pyfiglet import Figlet
import time
import pandas as pd


df = pd.DataFrame({'a': [1,2,3],
                    'b': [3,4,5]})
f = Figlet(font='banner3-D')
colors = ['yellow', 'red', 'green', 'blue']
print('Your original data is')
print(df)
for i, color in enumerate(colors):
    print(colored(f.renderText(f'Model {i+1}'), color))
    print('****************Training****************')
    time.sleep(2)
    print('Output is')
    print(df.multiply(i))
    print('****************Complete****************')
