import os
from dataclasses import dataclass, fields

import pandas as pd
import numpy as np

from .constants import (
    FOLDER_DATA_NORMALIZED
)

@dataclass
class AARCCollection:
    """A dataclass for the AARC data.
    """
    persons: pd.DataFrame
    departments: pd.DataFrame
    taxonomies: pd.DataFrame
    fields: pd.DataFrame
    areas: pd.DataFrame
    umbrellas: pd.DataFrame
    institutions: pd.DataFrame

    # Relationships
    appointments: pd.DataFrame
    department_taxonomies: pd.DataFrame

    @classmethod
    def from_collection_folder(
            cls,
            file_path: str = FOLDER_DATA_NORMALIZED
    ):
        aarc_collection = {}
        for filename in os.listdir(file_path):
            if filename.endswith(".csv"):
                field = filename[:-4]
                if field not in [f.name for f in fields(cls)]:
                    continue
                df = pd.read_csv(os.path.join(file_path, filename), index_col=0).convert_dtypes()
                aarc_collection[field] = df
        return cls(**aarc_collection)

def _normalize_unidentified_data(
        df_data: pd.DataFrame,
        key: str,
        verbose: bool) -> pd.DataFrame:
    df_atomic = df_data[key].dropna().unique()
    df_atomic = pd.DataFrame(
        np.arange(len(df_atomic)),
        index=df_atomic,
        columns=[f"{key}Id"],
        dtype=pd.Int64Dtype()
    )
    if verbose:
        print(f"\t{key} count: {df_atomic.shape[0]}")
    return df_atomic

def normalize_data(
        df_data: pd.DataFrame,
        verbose: bool = True
) -> AARCCollection:
    df_persons = df_data.groupby("PersonId")\
        .aggregate({
            "Gender": "first",
            "DegreeYear": "first",
            "PersonName": "first",
            "DegreeInstitutionId": "first",
    })
    if verbose:
        print(f"\tFaculty count: {df_persons.shape[0]}")


    df_departments = df_data.groupby("DepartmentId")\
        .aggregate({
            "DepartmentName": "first"
    })
    if verbose:
        print(f"\tDepartments count: {df_departments.shape[0]}")

    df_institutions = df_data.groupby("InstitutionId")\
        .aggregate({
            "InstitutionName": "first"
    })
    if verbose:
        print(f"\tInstitutions count: {df_institutions.shape[0]}")

    # Get units of domains, areas and fields
    # from unique values in the data, create artificial IDs
    # and handle missing values as `pd.NA`
    df_taxonomies, df_domains, df_areas, df_fields = [
        _normalize_unidentified_data(
            df_data=df_data,
            key=key,
            verbose=verbose)
        for key in ["Taxonomy", "Umbrella", "Area", "Field"]
    ]

    df_departments_taxonomies = df_data[
        ["DepartmentId", "Taxonomy", "Field", "Umbrella", "Area"]]\
        .drop_duplicates()\
        .join(
            df_fields,
            on="Field")\
        .join(
            df_areas,
            on="Area")\
        .join(
            df_domains,
            on="Umbrella")\
        .drop(columns=["Area", "Umbrella", "Field"])
    if verbose:
        print(("\tDepartment x Taxonomy x Field x Area x Umbrella count: "
               f"{df_departments_taxonomies.shape[0]}"))

    df_appointments = df_data\
        .groupby(
            ["PersonId", "Year", "DepartmentId", "InstitutionId"],
            dropna=False)\
        .aggregate({
            "Rank": "first",
            "PrimaryAppointment": "first"
        })\
        .reset_index()

    return AARCCollection(
        persons=df_persons,
        departments=df_departments,
        taxonomies=df_taxonomies,
        institutions=df_institutions,
        fields=pd.Series(
            df_fields.index.values,
            index=df_fields["FieldId"],
            name="Field"),
        areas=pd.Series(
            df_areas.index.values,
            index=df_areas["AreaId"],
            name="Area"),
        umbrellas=pd.Series(
            df_domains.index.values,
            index=df_domains["UmbrellaId"],
            name="Umbrella"),
        appointments=df_appointments.set_index(
            ["PersonId", "Year", "DepartmentId", "InstitutionId"]),
        department_taxonomies=df_departments_taxonomies,
    )
