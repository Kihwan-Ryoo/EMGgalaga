#include <msp430x16x.h>

int adc1,adc2,adc3,adc4,adc5,adc6;
unsigned char Packet[13];

void ReadAdc12 (void);      // Read data from internal 12 bits ADC

void main(void)
{
    unsigned int i;

//  Set basic clock and timer
   WDTCTL = WDTPW + WDTHOLD;                 // Stop WDT
   BCSCTL1 &= ~XT2OFF;                       // // XT2 on
   do{
           IFG1 &=~OFIFG;                    // Clear oscillator flag
           for(i=0;i<0xFF;i++);              // Delay for OSC to stabilize
   }while((IFG1&OFIFG));  

   BCSCTL2 |= SELM_2;                        // MCLK =XT2CLK=6Mhz
   BCSCTL2 |= SELS;                          // SMCLK=XT2CLK=6Mhz

// Set Port  
   P3SEL = BIT4|BIT5;                            // P3.4,5 = USART0 TXD/RXD
   P6SEL = 0x3f;   P6DIR=0x3f;   P6OUT=0x00;
//Set UART0  
   ME1 |= UTXE0 + URXE0;                     // Enable USART0 TXD/RXD
   UCTL0 |= CHAR;                            // 8-bit character
   UTCTL0 |= SSEL0|SSEL1;                    // UCLK= SMCLK
   UBR00 = 0x71;                             // 6MHz 19600=> baud rate = 9600
   UBR10 = 0x02;                             // 6MHz 9600
   UMCTL0 = 0x00;                            // 6MHz 9600 modulation
   UCTL0 &= ~SWRST;                          // Initialize USART state machine

//  Set 12bit internal ADC
   ADC12CTL0 = ADC12ON | REFON | REF2_5V;                // ADC on, 2.5 V reference on
   ADC12CTL0 |= MSC;                                     // multiple sample and conversion
   ADC12CTL1 = ADC12SSEL_3 | ADC12DIV_7 | CONSEQ_1;      // SMCLK, /8, sequence of channels (01)- ����ä�ο� ���� ADC�ϱ� ����
   ADC12CTL1 |= SHP;      

   ADC12MCTL0 = SREF_0 | INCH_0; //TP1
   ADC12MCTL1 = SREF_0 | INCH_1; //TP2
   ADC12MCTL2 = SREF_0 | INCH_2;  //TP3
   ADC12MCTL3 = SREF_0 | INCH_3 | EOS; //TP4 input == sequence �� ������
//   ADC12MCTL4 = SREF_0 | INCH_4;
//   ADC12MCTL5 = SREF_0 | INCH_5 | EOS;

   ADC12CTL0 |= ENC;                                     // enable conversion

//  SetTimerA
   TACTL=TASSEL_2+MC_1;                        // clock source and mode(UP) select
   TACCTL0=CCIE;
   TACCR0 = 1200;                   //  6M/1200=5kHz <- sampling rate

  
  _BIS_SR(LPM0_bits + GIE);                 // Enter LPM0 w/ interrupt  

}


#pragma vector = TIMERA0_VECTOR
__interrupt void TimerA0_interrupt()
{
   ReadAdc12();

   Packet[0]=(unsigned char)0x81; //header1, 1Byte (�޼�)
   __no_operation();

   Packet[1]=(unsigned char)(adc1>>7)&0x7F; //�޼�1
   Packet[2]=(unsigned char)adc1&0x7F;
  
   Packet[3]=(unsigned char)(adc2>>7)&0x7F; //�޼�2
   Packet[4]=(unsigned char)adc2&0x7F;

   Packet[5] = 0;
   Packet[6] = 0; //dummu byte

   Packet[7] = (unsigned char)0x81; //header2, 1Byte (������)
   __no_operation();

   Packet[8] = (unsigned char)(adc3 >> 7) & 0x7F; //������1
   Packet[9] = (unsigned char)adc3 & 0x7F;

   Packet[10] = (unsigned char)(adc4 >> 7) & 0x7F; //������2
   Packet[11] = (unsigned char)adc4 & 0x7F;

   Packet[12] = 0;
   Packet[13] = 0; //dummu byte

   //header1�� ����
   for (int j = 0; j < 7; j++) {
      while (!(IFG1 & UTXIFG0));                // USART0 TX buffer ready? IFGx[UTXIFG] = 1�̸� �۽��� �� ������ �ǹ�.
      TXBUF0 = Packet[j];
   }
   //header2�� ����
   for (int k = 7; k < 14; k++) {
      while (!(IFG1 & UTXIFG0));                // USART0 TX buffer ready?
      TXBUF0 = Packet[k];
   }
}

void ReadAdc12 (void)
{

   /*4096 = 2^12�ػ� / 9000= (4.5V->4500mV )*2  => ��ü �Է� ���� ������ ADC�ػ󵵷� ���� ��.
      -4500�� -4500~+4500������ ���Բ�, 7000�� ���Ƿ� ����� ���� ��ȣ��Ʈ�� �Ⱦ������Բ�
       �ø��� ������� �״�� �����ϸ� ��ȣ��Ʈ�� ������*/

   // read ADC12 result from ADC12 conversion memory
   // start conversion and store result without CPU intervention
   adc1 = (int)( (long)ADC12MEM0 * 9000 / 4096) -4500+7000;             // adc0 voltage in [mV]
   adc2 = (int)( (long)ADC12MEM1 * 9000 / 4096) -4500+7000; 
   adc3 = (int)( (long)ADC12MEM2 * 9000 / 4096) -4500+7000; 
   adc4 = (int)( (long)ADC12MEM3 * 9000 / 4096) -4500+7000; 
   //adc5 = (int)( (long)ADC12MEM4 * 9000 / 4096) -4500+7000; 
   //adc6 = (int)( (long)ADC12MEM5 * 9000 / 4096) -4500+7000; 
            
   ADC12CTL0|=ADC12SC;                                               // start conversion

    /* ADC=0 : -4.5V, ADC=4095 : 4.5V */ 
}