Log:

1. Improved the ILS_solver:

↑ Previously: compulsively insert a most interesting visit with relatively low cost: ratio = 𝑼_𝑖 / travel cost
↓ Now: no compulsory visit, totally based on utility gain. The combinations where travel impedance is big will result in empty path predicted.

Now the total utility gained is significant.

--- File information:
1. 较为之前生成的
2. Initialization objective values fixed shape: fixed shape at 7, scale is flexible.
Both are evaluated under the improved ILS_solver

↑ last modified on 11:22 Jan.24 2020