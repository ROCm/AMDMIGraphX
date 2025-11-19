// Test for EmptyDoWhileStatement check

void test_empty_do_while()
{
    int i = 0;
    // cppcheck-suppress emptyDoWhileStatement
    do
    {
    } while(i++ < 5);
}

void test_do_while_with_statement()
{
    int i = 0;
    do
    {
        i++;
    } while(i < 5);
}

void test_do_while_with_multiple_statements()
{
    int i = 0;
    do
    {
        i++;
        if(i == 3)
            break;
    } while(i < 10);
}
