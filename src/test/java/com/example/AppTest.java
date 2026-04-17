package com.example;

import junit.framework.TestCase;

public class AppTest extends TestCase {

    public void testAdd() {
        App app = new App();
        int result = app.add(2, 3);
        assertEquals(5, result);
    }

    public void testSubtract() {
        App app = new App();
        int result = app.subtract(5, 3);
        assertEquals(2, result);
    }

    public void testIsEven() {
        App app = new App();
        assertTrue(app.isEven(4));
        assertFalse(app.isEven(5));
    }
}
