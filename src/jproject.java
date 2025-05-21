import org.openqa.selenium.*;
// import org.openqa.selenium.By.ByXPath;
import org.openqa.selenium.chrome.*;
// import org.openqa.selenium.support.ui.ExpectedConditions;
// import org.openqa.selenium.support.ui.WebDriverWait;
import java.util.*;

import java.time.Duration; 
// import io.github.cdimascio.dotenv.Dotenv;

// enter email password jobname Phone number
public class jproject {      
String EMAIL="";
String PASSWORD="";
String jobname="+";
String PNO="";
int maxapplications=2;



public void Driver(){
System.setProperty("webdriver.chrome.driver", "chromedriver-win64/chromedriver-win64/chromedriver.exe");
    WebDriver driver = new ChromeDriver();
    // JavascriptExecutor js = (JavascriptExecutor) driver;
    driver.get("https://www.linkedin.com");
    Random r = new Random();

    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(30));
    System.out.println("i");
    WebElement signin= driver.findElement(By.cssSelector("a[data-tracking-control-name='guest_homepage-basic_nav-header-signin']"));
    signin.click();
    
    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(30));
    WebElement  email=driver.findElement(By.id("username"));
    email.sendKeys(EMAIL);
    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
    WebElement  password=driver.findElement(By.id("password"));
    password.sendKeys(PASSWORD);
    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(20));
    WebElement submit=driver.findElement(By.cssSelector("button[data-litms-control-urn='login-submit']"));
    submit.click();

    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(60));
    WebElement searchele=driver.findElement(By.cssSelector("input[data-view-name='search-global-typeahead-input']"));
    searchele.sendKeys(jobname);
    searchele.sendKeys(Keys.ENTER);
    
    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(20));
    WebElement expand=driver.findElement(By.xpath("//button[contains(@class, 'artdeco-pill--choice') and .//text()='Jobs']"));
    expand.click();

    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(40));
    // List<WebElement> jobl = driver.findElements( By.cssSelector("a.job-card-job-posting-card-wrapper__card-link"));
    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(37));
    
    int applicationsSent = 0;
    while(applicationsSent<=maxapplications){
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(50));
        List<WebElement> job1 = driver.findElements( By.cssSelector("a.job-card-job-posting-card-wrapper__card-link"));
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(60));
        int index = r.nextInt(job1.size());
        WebElement job = driver.findElements(By.cssSelector("a.job-card-job-posting-card-wrapper__card-link")).get(index);
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(17));
        job.click();
        try {
                    System.out.println("work");
                    WebElement apply = driver.findElement(By.xpath("//button[contains(@class, 'jobs-apply-button')]" +
                    "//span[contains(@class, 'artdeco-button__text') and normalize-space()='Easy Apply']" +
                    "/ancestor::button[1]"));
                    apply.click();
                    System.out.println('j');   
                    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(5));
                    WebElement pno=driver.findElement( By.xpath("//input[contains(@class, 'artdeco-text-input--input') and contains(@id, 'phoneNumber-nationalNumber')]"));
                    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
                    pno.sendKeys(PNO);
                    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(5));
                    WebElement next=driver.findElement(By.xpath("//button[@aria-label='Continue to next step' and .//span[text()='Next']]"));
                    next.click();
                    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(27));
                    WebElement next2=driver.findElement(By.xpath("//button[@aria-label='Continue to next step' and .//span[text()='Next']]"));
                    next2.click();
                    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(30));
                    WebElement close=driver.findElement(By.cssSelector("svg[data-test-icon='close-medium']"));
                    close.click();
                    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
                    WebElement fclose=driver.findElement(By.cssSelector("button[data-test-dialog-secondary-btn]"));
                    fclose.click();
                    applicationsSent+=1;
                    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(35));
            } 
            catch (Exception e){
                    System.out.println(e);
                    System.out.println("missed job application");
                    driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(60));
            }

    }
    driver.close();
}

    // // int applicationsSent = 0;
    // for(WebElement job:jobl){
    //     if(applicationsSent <= maxapplications){
    //         if(applicationsSent!=0){
    //         System.out.println("it");
    //         driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(17));
    //         job.click();
    //         }
    //         driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
    //         try {
    //             System.out.println("work");
    //             WebElement apply = driver.findElement(By.xpath("//button[contains(@class, 'jobs-apply-button')]" +
    //             "//span[contains(@class, 'artdeco-button__text') and normalize-space()='Easy Apply']" +
    //             "/ancestor::button[1]"));
    //             apply.click();
    //             System.out.println('j');   
    //             driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(5));
    //             WebElement pno=driver.findElement( By.xpath("//input[contains(@class, 'artdeco-text-input--input') and contains(@id, 'phoneNumber-nationalNumber')]"));
    //             driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
    //             pno.sendKeys(PNO);
    //             driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(5));
    //             WebElement next=driver.findElement(By.xpath("//button[@aria-label='Continue to next step' and .//span[text()='Next']]"));
    //             next.click();
    //             driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(27));
    //             WebElement next2=driver.findElement(By.xpath("//button[@aria-label='Continue to next step' and .//span[text()='Next']]"));
    //             next2.click();
    //             driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(30));
    //             WebElement close=driver.findElement(By.cssSelector("svg[data-test-icon='close-medium']"));
    //             close.click();
    //             driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
    //             WebElement fclose=driver.findElement(By.cssSelector("button[data-test-dialog-secondary-btn]"));
    //             fclose.click();
    //             applicationsSent+=1;
    //             driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(35));
    //         } catch (Exception e){
    //             System.out.println(e);
    //             System.out.println("missed job application");
    //             applicationsSent+=1;
    //         }
    //         }         
    //     }
    //     // driver.close();
    // }  
    


 public static void main(String[] args) {
  jproject obj=new jproject();
  obj.Driver();
 }
}
