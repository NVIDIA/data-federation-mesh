# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the PolymorphicBaseModel class."""

from typing import List

from pydantic import BaseModel

from nv_dfm_core.api.pydantic import PolymorphicBaseModel


################################
# Test models
class Animal(PolymorphicBaseModel):
    name: str


class Zoo(BaseModel):
    animals: List[Animal]


class Dog(Animal):
    dfm_class_name: str = "test_polymorphic_base_model.Dog"
    breed: str


class Cat(Animal):
    dfm_class_name: str = "test_polymorphic_base_model.Cat"
    color: str


def test_polymorphic_deserialization():
    zoo1 = Zoo(
        animals=[
            Dog(name="Buddy", breed="Labrador"),
            Cat(name="Whiskers", color="Black"),
        ]
    )
    json = zoo1.model_dump_json()
    zoo2 = Zoo.model_validate_json(json)
    assert isinstance(zoo2.animals[0], Dog)
    assert isinstance(zoo2.animals[1], Cat)


################################
# Test models with discriminator


class Vehicle(PolymorphicBaseModel):
    @classmethod
    def _discriminator_name(cls) -> str:
        return "vehicle_type"

    vehicle_type: str
    name: str


class Garage(BaseModel):
    vehicles: List[Vehicle]


class Bike(Vehicle):
    vehicle_type: str = "test_polymorphic_base_model.Bike"
    gears: int


class Car(Vehicle):
    vehicle_type: str = "test_polymorphic_base_model.Car"
    color: str


def test_polymorphic_deserialization_with_discriminator():
    garage1 = Garage(
        vehicles=[Bike(name="Bike", gears=10), Car(name="Car", color="Red")]
    )
    json = garage1.model_dump_json()
    garage2 = Garage.model_validate_json(json)
    assert isinstance(garage2.vehicles[0], Bike)
    assert isinstance(garage2.vehicles[1], Car)
