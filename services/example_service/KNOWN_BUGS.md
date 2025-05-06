# Known Bugs in Example Service

This document lists the intentional bugs introduced in the example service for demonstration purposes.

## Bug #1: Missing Error Handling for Non-existent IDs

**File:** `app.py`  
**Function:** `get_todo()`  
**Line:** ~76-79  
**Description:** The endpoint doesn't check if the requested todo_id exists in the database before attempting to access it, causing a KeyError exception.  
**Expected Fix:** Add a check for todo_id existence and raise a proper HTTPException with a 404 status code.

## Bug #2: Missing Field Initialization

**File:** `app.py`  
**Function:** `create_todo()`  
**Line:** ~67-70  
**Description:** The 'completed' field is not initialized when creating a new todo item, which will cause issues when clients expect this field to exist.  
**Expected Fix:** Add initialization of the 'completed' field to False when creating a new todo.

## Bug #3: Incorrect Dict Parameter

**File:** `app.py`  
**Function:** `update_todo()`  
**Line:** ~90-92  
**Description:** The dict() method is called without the exclude_unset parameter, causing all fields to be included in the update data even if not provided in the request. This can overwrite existing values with None.  
**Expected Fix:** Use todo.dict(exclude_unset=True) to only include fields that were explicitly set in the request.

## Bug #4: Unsafe List Slicing

**File:** `app.py`  
**Function:** `get_todos()`  
**Line:** ~58-61  
**Description:** The endpoint doesn't check if the skip and limit parameters would result in valid slice indices before performing the slice operation.  
**Expected Fix:** Add bounds checking to ensure skip doesn't exceed the length of the list and limit the slice to the available items.

## Bug #5: Unsafe Environment Variable Conversion

**File:** `app.py`  
**Function:** Main execution block  
**Line:** ~115-117  
**Description:** The code doesn't handle the case where the PORT environment variable contains a non-integer value, which would cause a ValueError.  
**Expected Fix:** Add a try/except block to handle potential ValueError when converting PORT to an integer.