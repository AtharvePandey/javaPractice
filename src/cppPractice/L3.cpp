// access modifiers


/*
    video questions below:
        1.) difference between private and protected members in cpp
            - both need public methods to be modified, but the difference comes when we look at child classes
            - child classes can't access private members (method or variable) of their parent (unless via public methods)
            - but if those variables are protected, then the child class can directly access it
        2.) why do we have private member variables
            - because depending on class, some things should be modified after checks/verifications, so those things are private
            - will get manipulated by public programmer accessible methods.

*/