import pytest
from torchregress.causal.dr import dr_ate
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression

def test_coverage():
    print("Testing")
