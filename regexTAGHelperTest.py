# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 21:03:57 2016

@author: darks
"""

import re

import TAGHelper

S = [r'\o [\a [\o] R ]',r'\o [R \o]',r'\a [\*o [\a] "r"]',\
     r'R [\a [\*o ["r"] "s"]]',r'R ["r" "s"]',r'\a ["r" \a [\a]]',\
     r'\a [\a \a ["s"]]',r'\a ["s" \a]',r'\a [\*o [\a] \a [R]]',\
     r'\o [\a [\o "r"]]']

rootPattern  = r"\s*((?:\\\*?)?)(\w+)\s*\[(.*)\]"
pattern = r'\s*((?:\\\*?)?\w+|"[^"]*"|\[|\])'

root = re.findall(rootPattern,S[2])[0]
nodes = re.findall(pattern,root[-1])

# -----------------------------------------------------------------------------
d = TAGHelper.TAGTokenize(S[2])

sp = re.findall(pattern,S[2])