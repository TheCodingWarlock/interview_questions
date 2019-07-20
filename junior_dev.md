# Junior Developer Interview Questions

Common Junior dev Questions curated from the internet.<br>
*Disclaimer* I'm not in HR.
Sources
1. [interview-questions-for-graduatejunior-software-developers](https://ilovefoobar.wordpress.com/2012/12/15/interview-questions-for-graduatejunior-software-developers/)
2. [https://www.fullstack.cafe/](https://www.fullstack.cafe/)

<br>

## Questions
<details><summary><b>Tell us  about yourself.</b></summary>
<p>
one of the best strategy is to focus on this employer and your fit for this job. No body wants to know about your 10 cats.
</p>
</details>
<details><summary><b>Other than study and programming, what you like to do during your free time.
> 

<details><summary><b>What is difference between overwriting and overloading in OOP.</b></summary>
<p>

> * Overloading* occurs when two or more methods in one class have the same method name but different parameters.<br>
> * Overriding* means having two methods with the same method name and parameters (i.e., method signature). One of the methods is in the parent class and the other is in the child class. Overriding allows a child class to provide a specific implementation of a method that is already provided its parent class.

</p>
</details>

<details><summary><b>Tell us about your experience while working in team.</b></summary>
<p>

> Aim of this question is to findout if you're a team play. Don't imply that without you the team wouldn't make it also be careful not to come across as the weakest link in the team. Mention your achievemnts personal and also as a team.

</p>
</details>
<details><summary><b>How do you manage conflicts in a group assignments.
> * .
</p>
</details>

<details><summary><b>Write a sql query to join two tables in database.</b></summary>
<p>

> * (INNER) JOIN: Returns records that have matching values in both tables<br><br>
`SELECT column_name(s)
FROM table1
INNER JOIN table2
ON table1.column_name = table2.column_name;`<br><br>
> * LEFT (OUTER) JOIN: Returns all records from the left table, and the matched records from the right table<br><br>
`SELECT column_name(s)
FROM table1
LEFT JOIN table2
ON table1.column_name = table2.column_name;`<br><br>
> * RIGHT (OUTER) JOIN: Returns all records from the right table, and the matched records from the left table <br><br>
`SELECT column_name(s)
FROM table1
RIGHT JOIN table2
ON table1.column_name = table2.column_name;`<br><br>
> * FULL (OUTER) JOIN: Returns all records when there is a match in either left or right table <br><br>
`SELECT column_name(s)
FROM table1
FULL OUTER JOIN table2
ON table1.column_name = table2.column_name
WHERE condition;`<br><br>

</p>
</details>
<details><summary><b>Imagine you have two array a = [1,2,3,4,5] and b =[3,2,9,3,7], write a program to find out common elements in both array.</b></summary>


```
a = [1, 2, 3, 4, 5]
b = [3, 2, 9, 3, 7]
temp = []
for i in range(len(b)):
    if a[i] in b:
        temp.append(a[i])

print(temp)

```


</details>


<details><summary><b>( Related to question above.) Can you write this without using for loop? </b></summary>

```

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        print(a_set & b_set)
    else:
        print("No common elements")
    a = [1, 2, 3, 4, 5]
    b = [3, 2, 9, 3, 7]
    common_member(a, b)
```


   </p>
</details>


<details><summary><b> If i sort those arrays will it make any difference in your code? Can you write better code if arrays are sorted? </b></summary>
   
   >	Time complexity will be same.		
   
</details>

<details><summary><b> What is different between ArrayList and Set.</b></summary>

> List is a type of ordered collection that maintains the elements in insertion order while Set is a type of unordered collection so elements are not maintained any order.

> List allows duplicates while Set doesn't allow duplicate elements . All the elements of a Set should be unique if you try to insert the duplicate element in Set it would replace the existing value.

> List permits any number of null values in its collection while Set permits only one null value in its collection.

> New methods are defined inside List interface . But, no new methods are defined inside Set interface, so we have to use Collection interface methods only with Set subclasses .

> List can be inserted in in both forward direction and backward direction using Listiterator while Set can be traversed only in forward direction with the help of iterator 
 
</details>

<details><summary><b> Write sql query to find out total number of item sold 
for certain product and list them in descending order. </b></summary>

```
SELECT ProductID, count(*) AS NumSales FROM Orders GROUP BY ProductID DESC;
```

</details>


<details><summary><b>What is use of index in database? Give us example of columns that should be indexed. </b></summary>

>  Indexes are used to quickly locate data without having to search every row in a database table every time a database table is accessed. You can use a combination of columns. you can index UPPER(LastName)

</details>


<details><summary><b> What is use of foreign key in database? </b></summary>
 
 > A foreign key is a column or group of columns in a relational database table that provides a link between data in two tables. It acts as a cross-reference between tables because it references the primary key of another table, thereby establishing a link between them

</details>


<details><summary><b> What is MVC pattern? </b></summary>

> an architectural pattern commonly used for developing user interfaces that divides an application into three interconnected parts. This is done to separate internal representations of information from the ways information is presented to and accepted from the user

</details>


<details><summary><b>Have you heard of any design pattern? Please name and explain couple of them. </b></summary>
 
 > [https://sourcemaking.com/design_patterns](https://sourcemaking.com/design_patterns)

</details>


<details><summary><b> What is data-structure? </b></summary>

> Data structure availability may vary by programming languages. Commonly available data structures are:
   * list,
   * arrays,
   * stack,
   * queues,
   * graph,
   * tree etc

</details>


<details><summary><b> What is algorithm?</b></summary>
  
  > Algorithm is a step by step procedure, which defines a set of instructions to be executed in certain order to get the desired output.
  
</details>


<details><summary><b>What is linear searching? </b></summary>
   <p>
		
   </p>
</details>


<details><summary><b> </b></summary>
   <p>
		
   </p>
</details>


<details><summary><b> </b></summary>
   <p>
		
   </p>
</details>



<details><summary><b> </b></summary>
   <p>
		
   </p>
</details>



<details><summary><b> </b></summary>
   <p>
		
   </p>
</details>



<details><summary><b> </b></summary>
   <p>
		
   </p>
</details>


<details><summary><b> </b></summary>
   <p>
		
   </p>
</details>