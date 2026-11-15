# any errors in here:

<pasted_content id="8995">
CR
_Exported on 11/11/2026 at 15:21:31 GMT-6 from Claude via WayLog_


**User**

any errors in here:

<pasted_content id="8995">
CREATE OR REPLACE PROCEDURE Invoice(customerID  Customer.C_ID%type, orderDate DATE)
AS
  customerName Customer.Name%type;
  
  TotalProductCost NUMBER := 0;
  ShippingCost NUMBER := 0;
  CURSOR InvoiceCur IS (SELECT P.P_Name AS I_NAME, L.Quantity AS I_QUANTITY, P.Price AS I_PRICE, P.Price * L.Quantity AS I_TOTAL  FROM PRODUCT P, LINE_ITEM L, PRODUCT_ORDER O WHERE L.PRODUCT_ID = P.P_ID AND O.ORDER_NO = L.ORDER_NO AND O.ORDER_DATE = orderDate AND O.Cust_ID = customerID);
  InvoiceRow InvoiceCur%rowtype; 
BEGIN
  SELECT NAME INTO customerName FROM CUSTOMER WHERE CUSTOMER.C_ID = customerID;
  DBMS_OUTPUT.PUT_LINE(orderDate || ' Invoice for ' || customerID || ': ' || customerName);
  FOR InvoiceRow IN InvoiceCur LOOP
    DBMS_OUTPUT.PUT_LINE(InvoiceRow.I_NAME || TO_CHAR(InvoiceRow.I_QUANTITY) || TO_CHAR(InvoiceRow.I_PRICE) || TO_CHAR(InvoiceRow.I_TOTAL));
    TotalProductCost := InvoiceRow.I_TOTAL + TotalProductCost;
  END LOOP;
  
  IF TotalProductCost > 35 THEN
    ShippingCost := 10;
    
  END IF;
  DBMS_OUTPUT.PUT_LINE('Total Product Cost: ' || TotalProductCost);
  DBMS_OUTPUT.PUT_LINE('Shipping Cost: ' || ShippingCost);
  DBMS_OUTPUT.PUT_LINE('Total Due: '|| TotalProductCost + ShippingCost);
  

END;
/
call Invoice('C07', '30-JAN-25');
</pasted_content id="8995">

---


**Claude**

The procedure compiles, but one line will fail at runtime. There are also a few things you should check.

## Will fail at runtime

**The "Total Due" line.** In PL/SQL, `||` and `+` have the same precedence and are evaluated left to right. So this:
```sql
'Total Due: ' || TotalProductCost + ShippingCost
```
runs as `('Total Due: ' || TotalProductCost) + ShippingCost`. That tries to add a number to the string `'Total Due: 45'` and fails with **ORA-06502 / ORA-01722 (invalid number)**. Add parentheses:
```sql
DBMS_OUTPUT.PUT_LINE('Total Due: ' || (TotalProductCost + ShippingCost));
```

## Worth checking

1. **The shipping rule may be backwards.** As written, orders **over** $35 pay $10 shipping and smaller orders ship free. The usual rule is free shipping over a threshold. Check your lab spec. If it's the usual rule, use:
   ```sql
   IF TotalProductCost <= 35 THEN ShippingCost := 10; END IF;
   ```
2. **The date literal depends on session settings.** `'30-JAN-25'` only works if your session's date format is `DD-MON-RR`. A safer call is:
   ```sql
   CALL Invoice('C07', DATE '2025-01-30');
   ```
   Also, if `ORDER_DATE` stores a time as well as a date, `O.ORDER_DATE = orderDate` won't match. In that case compare `TRUNC(O.ORDER_DATE) = orderDate`.
3. **The column names in `%type` must match your table.** You use `C_ID` and `P_Name` elsewhere, but `Customer.Name%type` and `SELECT NAME` assume the column is called `NAME`. If it's actually `C_NAME`, the procedure won't compile.
4. **You need `SET SERVEROUTPUT ON` before the `CALL`**, or nothing will print.
5. **An unknown customer ID raises `NO_DATA_FOUND`** in the `SELECT ... INTO`. That's fine for a lab, but you could add an `EXCEPTION` block to handle it.

## Minor
- **`InvoiceRow InvoiceCur%rowtype;` isn't used.** The `FOR InvoiceRow IN InvoiceCur` loop declares its own loop record, which hides this one. You can delete the declaration.
- **The item lines have no separators,** so values run together (e.g. `Widget25.0050.00`). Add some spacing:
  ```sql
  DBMS_OUTPUT.PUT_LINE(InvoiceRow.I_NAME || '  ' || InvoiceRow.I_QUANTITY || '  ' ||
                       InvoiceRow.I_PRICE || '  ' || InvoiceRow.I_TOTAL);
  ```

---


**User**

getting numeric or value error

---


**Claude**

You're almost certainly hitting the "Total Due" line. Has that been changed to this yet?

```sql
DBMS_OUTPUT.PUT_LINE('Total Due: ' || (TotalProductCost + ShippingCost));
```

Without the parentheses, Oracle first joins the label and the number into a string like `'Total Due: 45'`. Then it tries to add `ShippingCost` to that string, which raises `ORA-06502: numeric or value error`. The invoice lines above it can print fine before it fails, so if you see output and then the error, this is the cause.

If it's already fixed and you still get the error, look at the line number in the message (e.g. `ORA-06512: at "INVOICE", line 22`). Two other likely causes:

- **The error appears before anything prints:** `C_ID` might be a `NUMBER` column. Passing `'C07'` would then fail to convert. Check with `DESC CUSTOMER;`.
- **The error points at the `SELECT ... INTO` line:** `customerName` might not match the type of the column you're selecting. Make sure `Customer.Name%type` names the same column as `SELECT NAME`.

If none of these fit, paste the full error message and I'll pin down the line.

---


**User**

date not valid for month specified

---


**Claude**

That's `ORA-01839`. Oracle is converting a string to a date and finding a day that doesn't exist in its month, like `30-FEB` or `31-APR`. `30-JAN` itself is valid, so the bad date is probably somewhere else. There are two likely causes.

**1. `ORDER_DATE` is a text column with a bad value in it (most likely).** If `PRODUCT_ORDER.ORDER_DATE` is `VARCHAR2` instead of `DATE`, then `O.ORDER_DATE = orderDate` makes Oracle convert every row's text to a date. One impossible date in your data breaks the whole query. Check both:

```sql
DESC PRODUCT_ORDER;
SELECT ORDER_NO, ORDER_DATE FROM PRODUCT_ORDER;
```

Look for something like `29-FEB-25` (2025 isn't a leap year), `31-APR-..`, `31-JUN-..`, `31-SEP-..` or `31-NOV-..`. Fix that row, or better, make the column a real `DATE`.

**2. The call's literal is being read with a different date format.** Your session's format might not be `DD-MON-RR`. Pass the date without depending on that format:

```sql
CALL Invoice('C07', DATE '2025-01-30');
```

If that call alone fixes it, the session format was the problem. If the error still happens, it's cause 1, and the `SELECT` above will show you the bad row.

---
