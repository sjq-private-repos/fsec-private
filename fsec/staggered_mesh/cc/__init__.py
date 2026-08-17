#!/usr/bin/env python
# Copyright 2026 Stephen Quiton. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This package was written independently by Stephen Quiton.  It credits Xin
# Xing's experimental KRCCD code and local commit e37974c, Xin Xing and Lin
# Lin for the method in Phys. Rev. X 14, 011059, and the PySCF developers,
# including J. D. McClain and T. Berkelbach, for the upstream CCSD code.

"""Restricted periodic CCD with the Phys. Rev. X Madelung corrections."""

from .krccd import KCCD, KRCCD

__all__ = ["KRCCD", "KCCD"]

